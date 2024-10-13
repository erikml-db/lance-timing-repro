use std::fs::OpenOptions;
use std::io::Write;

use arrow::datatypes::Float32Type;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use clap::Parser;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::{WriteMode, WriteParams};
use lance::Dataset;
use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

use lance::index::vector::VectorIndexParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::{DatasetIndexExt, IndexType};
use lance_linalg::distance::MetricType;

use std::time::Instant;

const URI: &str = "./data";
const NUM_BITS: usize = 1;

#[derive(Parser, Debug)]
struct Args {
    /// The index version to use: v1 or v3
    #[clap(long, default_value = "v1")]
    index_version: String,

    #[clap(long, default_value_t = 500_000)]
    num_datapoints: usize,

    #[clap(long, default_value_t = 512)]
    num_centroids: usize,

    #[clap(long, default_value = "default")]
    query_type: String,

    #[clap(long, default_value_t = 768)]
    dim: usize,

    #[clap(long, default_value_t = 256)]
    sample_rate: usize,

    #[clap(long, default_value_t = 50)]
    max_iters: usize,
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args = Args::parse();

    let num_subvectors: usize = args.dim / 16;

    println!("Creating dataset");
    let mut dataset = generate_random_dataset(args.num_datapoints, args.dim).await;
    println!("Creating index");
    let mut ivf_params = IvfBuildParams::new(args.num_centroids);
    ivf_params.max_iters = args.max_iters;
    ivf_params.sample_rate = args.sample_rate;
    let pq_params = PQBuildParams::new(num_subvectors, NUM_BITS);
    let metric_type = MetricType::L2;
    let params = match args.index_version.as_str() {
        "v1" => VectorIndexParams::with_ivf_pq_params(metric_type, ivf_params, pq_params),
        "v3" => VectorIndexParams::with_ivf_pq_params_v3(metric_type, ivf_params, pq_params),
        _ => {
            eprintln!("Invalid index version provided. Use 'v1' or 'v3'.");
            return;
        }
    };
    let file_name = format!("results_{}.txt", args.index_version);

    // Open or create a file for writing the results
    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&file_name)
        .unwrap();

    dataset
        .create_index(&["embedding"], IndexType::Vector, None, &params, true)
        .await
        .unwrap();
    dataset.validate().await.unwrap();

    let nprobes_values = vec![1, 10, 50, 100, 512];
    let refine_factor_values = vec![0, 1, 10, 100];

    println!("Running query");
    for &nprobes in &nprobes_values {
        for &refine_factor in &refine_factor_values {
            writeln!(
                file,
                "Running query with nprobes: {}, refine_factor: {}",
                nprobes, refine_factor
            )
            .unwrap();
            query_dataset(
                &dataset,
                nprobes,
                refine_factor,
                &mut file,
                &args.query_type,
                args.dim,
            )
            .await;
        }
    }
}

async fn query_dataset(
    dataset: &Dataset,
    nprobes: usize,
    refine_factor: u32,
    file: &mut std::fs::File,
    query_type: &str,
    dim: usize,
) {
    let mut scanner = dataset.scan();
    let default_query = vec![0.1; dim];
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let query = match query_type {
        "default" => default_query,
        "random" => (0..dim).map(|_| normal.sample(&mut rng) as f32).collect(),
        _ => panic!("query_type: {} is not defined.", query_type),
    };
    let f32_array = Float32Array::from(query);
    scanner.nearest("embedding", &f32_array, 10).unwrap();
    if nprobes > 0 {
        scanner.nprobs(nprobes);
    }
    if refine_factor > 0 {
        scanner.refine(refine_factor);
    }

    for _ in 0..10 {
        let start = Instant::now();
        scanner.try_into_batch().await.unwrap();
        let time = start.elapsed();

        writeln!(file, "Time: {:?}", time).unwrap();
    }
}

async fn generate_random_dataset(num_datapoints: usize, dim: usize) -> Dataset {
    let mut rng = thread_rng();

    let mut id_builder = Int32Array::builder(num_datapoints);
    for i in 0..num_datapoints {
        id_builder.append_value(i.try_into().unwrap());
    }
    let id_array = Arc::new(id_builder.finish()) as ArrayRef;

    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut vectors: Vec<Option<Vec<Option<f32>>>> = Vec::with_capacity(num_datapoints);
    for _ in 0..num_datapoints {
        let vector: Vec<Option<f32>> = (0..dim)
            .map(|_| Some(normal.sample(&mut rng) as f32))
            .collect();
        vectors.push(Some(vector));
    }
    let embedding_array =
        Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, dim as i32));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            true,
        ),
    ]));

    let batch = RecordBatch::try_new(schema.clone(), vec![id_array, embedding_array]);
    let batches = RecordBatchIterator::new([batch], schema.clone());
    let write_params = WriteParams {
        mode: WriteMode::Overwrite,
        ..Default::default()
    };
    Dataset::write(batches, URI, Some(write_params))
        .await
        .unwrap();
    DatasetBuilder::from_uri(URI)
        .with_index_cache_size(usize::MAX)
        .with_metadata_cache_size(usize::MAX)
        .load()
        .await
        .unwrap()
}
