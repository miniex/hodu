use super::batch::{Batch, DataItem};
use super::dataset::Dataset;
use hodu_core::{error::HoduResult, tensor::Tensor};

pub type CollateFn = fn(Vec<DataItem>) -> HoduResult<Batch>;

pub fn default_collate(items: Vec<DataItem>) -> HoduResult<Batch> {
    if items.is_empty() {
        return Err(hodu_core::error::HoduError::InternalError(
            "Cannot collate empty items".into(),
        ));
    }

    match &items[0] {
        DataItem::Single(_) => {
            let tensors: Vec<Tensor> = items
                .into_iter()
                .map(|item| match item {
                    DataItem::Single(t) => t,
                    _ => panic!("Inconsistent data types in batch"),
                })
                .collect();

            let refs: Vec<&Tensor> = tensors.iter().collect();
            let batched = Tensor::stack(&refs, 0)?;
            Ok(Batch::Single(batched))
        },
        DataItem::Pair(_, _) => {
            let mut data_tensors = Vec::new();
            let mut label_tensors = Vec::new();

            for item in items {
                match item {
                    DataItem::Pair(d, l) => {
                        data_tensors.push(d);
                        label_tensors.push(l);
                    },
                    _ => panic!("Inconsistent data types in batch"),
                }
            }

            let data_refs: Vec<&Tensor> = data_tensors.iter().collect();
            let label_refs: Vec<&Tensor> = label_tensors.iter().collect();

            let batched_data = Tensor::stack(&data_refs, 0)?;
            let batched_labels = Tensor::stack(&label_refs, 0)?;

            Ok(Batch::Pair(batched_data, batched_labels))
        },
        DataItem::Multiple(vec) => {
            let num_tensors = vec.len();
            let mut tensor_vecs: Vec<Vec<Tensor>> = vec![Vec::new(); num_tensors];

            for item in items {
                match item {
                    DataItem::Multiple(tensors) => {
                        if tensors.len() != num_tensors {
                            panic!("Inconsistent number of tensors in batch");
                        }
                        for (i, tensor) in tensors.into_iter().enumerate() {
                            tensor_vecs[i].push(tensor);
                        }
                    },
                    _ => panic!("Inconsistent data types in batch"),
                }
            }

            let mut batched = Vec::with_capacity(num_tensors);
            for tensor_vec in tensor_vecs {
                let refs: Vec<&Tensor> = tensor_vec.iter().collect();
                batched.push(Tensor::stack(&refs, 0)?);
            }

            Ok(Batch::Multiple(batched))
        },
    }
}

pub struct DataLoaderBuilder<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    collate_fn: CollateFn,
    seed: Option<u64>,
}

impl<D: Dataset> DataLoaderBuilder<D> {
    pub fn new(dataset: D) -> Self {
        Self {
            dataset,
            batch_size: 1,
            shuffle: false,
            drop_last: false,
            collate_fn: default_collate,
            seed: None,
        }
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    pub fn collate_fn(mut self, collate_fn: CollateFn) -> Self {
        self.collate_fn = collate_fn;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn build(self) -> DataLoader<D> {
        DataLoader::new_with_config(
            self.dataset,
            self.batch_size,
            self.shuffle,
            self.drop_last,
            self.collate_fn,
            self.seed,
        )
    }
}

pub struct DataLoader<D: Dataset> {
    dataset: D,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    collate_fn: CollateFn,
    indices: Vec<usize>,
    current_idx: usize,
    seed: Option<u64>,
    epoch: u64,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: D, batch_size: usize) -> Self {
        Self::new_with_config(dataset, batch_size, false, false, default_collate, None)
    }

    pub fn new_with_config(
        dataset: D,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        collate_fn: CollateFn,
        seed: Option<u64>,
    ) -> Self {
        let len = dataset.len();
        let indices: Vec<usize> = (0..len).collect();

        let mut loader = Self {
            dataset,
            batch_size,
            shuffle,
            drop_last,
            collate_fn,
            indices,
            current_idx: 0,
            seed,
            epoch: 0,
        };

        if shuffle {
            loader.shuffle_indices();
        }

        loader
    }

    fn shuffle_indices(&mut self) {
        use rand::prelude::*;
        use rand::rngs::SmallRng;

        let length = self.indices.len();
        let seed_base = self.seed.unwrap_or(0) + self.epoch;
        let mut rng = SmallRng::seed_from_u64(seed_base);

        for i in (1..length).rev() {
            let j = rng.random_range(0..=i);
            self.indices.swap(i, j);
        }
    }

    pub fn builder(dataset: D) -> DataLoaderBuilder<D> {
        DataLoaderBuilder::new(dataset)
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn num_batches(&self) -> usize {
        let len = self.dataset.len();
        if self.drop_last {
            len / self.batch_size
        } else {
            len.div_ceil(self.batch_size)
        }
    }

    pub fn reset(&mut self) {
        self.current_idx = 0;
        self.epoch += 1;
        if self.shuffle {
            self.shuffle_indices();
        }
    }

    pub fn has_next(&self) -> bool {
        if self.current_idx >= self.dataset.len() {
            return false;
        }

        if self.drop_last {
            let remaining = self.dataset.len() - self.current_idx;
            remaining >= self.batch_size
        } else {
            true
        }
    }

    pub fn next_batch(&mut self) -> HoduResult<Option<Batch>> {
        if !self.has_next() {
            return Ok(None);
        }

        let start_idx = self.current_idx;
        let end_idx = (start_idx + self.batch_size).min(self.dataset.len());

        let batch_indices = &self.indices[start_idx..end_idx];
        let mut items = Vec::with_capacity(batch_indices.len());

        for &idx in batch_indices {
            items.push(self.dataset.get(idx)?);
        }

        self.current_idx = end_idx;

        let batch = (self.collate_fn)(items)?;
        Ok(Some(batch))
    }

    pub fn iter_batches(&mut self) -> DataLoaderIterator<'_, D> {
        DataLoaderIterator { loader: self }
    }
}

pub struct DataLoaderIterator<'a, D: Dataset> {
    loader: &'a mut DataLoader<D>,
}

impl<'a, D: Dataset> Iterator for DataLoaderIterator<'a, D> {
    type Item = HoduResult<Batch>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.loader.next_batch() {
            Ok(Some(batch)) => Some(Ok(batch)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}
