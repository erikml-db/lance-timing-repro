use std::fs::OpenOptions;
use std::io::Write;

use lance::Dataset;
use lance::dataset::builder::DatasetBuilder;
use lance::dataset::{WriteMode, WriteParams};
use arrow::datatypes::Float32Type;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use std::sync::Arc;
use clap::Parser;

use arrow::array::{ArrayRef, Int32Array, Float32Array, FixedSizeListArray};
use arrow::datatypes::{DataType, Field, Schema};
use rand::thread_rng;
use rand_distr::{Distribution, Normal};

use lance_linalg::distance::MetricType;
use lance_index::{DatasetIndexExt, IndexType};
use lance_index::vector::pq::PQBuildParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance::index::vector::VectorIndexParams;

use std::time::Instant;

const DIM: usize = 768;
const URI: &str = "./data";
const NUM_SUBVECTORS: usize = 48;
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
}

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let args = Args::parse();

    let mut dataset = generate_random_dataset(args.num_datapoints).await;
    let params = match args.index_version.as_str() {
        "v1" => VectorIndexParams::ivf_pq(args.num_centroids, NUM_BITS as u8, NUM_SUBVECTORS, MetricType::L2, 50),
        "v3" => VectorIndexParams::with_ivf_pq_params_v3(
            MetricType::L2,
            IvfBuildParams::new(args.num_centroids),
            PQBuildParams::new(NUM_SUBVECTORS, NUM_BITS),
        ),
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

    dataset.create_index(&["embedding"], IndexType::Vector, None, &params, true)
            .await
            .unwrap();
    dataset.validate().await.unwrap();

    let nprobes_values = vec![1, 10, 50, 100, 512];
    let refine_factor_values = vec![0, 1, 10, 100];

    for &nprobes in &nprobes_values {
        for &refine_factor in &refine_factor_values {
            writeln!(file, "Running query with nprobes: {}, refine_factor: {}", nprobes, refine_factor).unwrap();
            query_dataset(&dataset, nprobes, refine_factor, &mut file).await;
        }
    }
}

async fn query_dataset(dataset: &Dataset, nprobes: usize, refine_factor: u32, file: &mut std::fs::File) {
    let mut scanner = dataset.scan();
    let query = vec![0.1; DIM];
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

async fn generate_random_dataset(num_datapoints: usize) -> Dataset {
    let mut rng = thread_rng();

    let mut id_builder = Int32Array::builder(num_datapoints);
    for i in 0..num_datapoints {
        id_builder.append_value(i.try_into().unwrap());
    }
    let id_array = Arc::new(id_builder.finish()) as ArrayRef;

    let normal = Normal::new(0.0, 1.0).unwrap();
    

    let mut vectors: Vec<Option<Vec<Option<f32>>>> = Vec::with_capacity(num_datapoints);
    for _ in 0..num_datapoints {
        let vector: Vec<Option<f32>> = (0..DIM)
            .map(|_| Some(normal.sample(&mut rng) as f32))
            .collect();
        vectors.push(Some(vector));
    }
    let embedding_array = Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, DIM as i32));

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("embedding", DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Float32, true)), DIM as i32), true),
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