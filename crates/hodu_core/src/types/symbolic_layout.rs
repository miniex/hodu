use super::dim::Dim;
use super::layout::Layout;
use super::symbolic_shape::SymbolicShape;
use smallvec::SmallVec;
use std::fmt;

/// Layout that can contain dynamic dimensions
///
/// Used during graph capture to represent layouts with data-dependent
/// dimensions. Strides are computed based on max_bound values for allocation.
#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SymbolicLayout {
    shape: SymbolicShape,
    /// Strides computed using max_bound for dynamic dimensions
    strides: SmallVec<[usize; 8]>,
    offset: usize,
}

impl SymbolicLayout {
    /// Creates a new symbolic layout with given shape and strides
    pub fn new(shape: SymbolicShape, strides: Vec<usize>) -> Self {
        Self {
            shape,
            strides: SmallVec::from_vec(strides),
            offset: 0,
        }
    }

    /// Creates a contiguous symbolic layout from a symbolic shape
    /// Strides are computed using max_bound for dynamic dimensions
    /// Returns None if any dynamic dimension lacks a max_bound
    pub fn from_shape(shape: &SymbolicShape) -> Option<Self> {
        // Ensure all dimensions have allocation sizes
        shape.max_size()?;

        let strides = Self::compute_strides(shape)?;
        Some(Self {
            shape: shape.clone(),
            strides,
            offset: 0,
        })
    }

    /// Creates a symbolic layout from a concrete Layout
    pub fn from_concrete(layout: &Layout) -> Self {
        Self {
            shape: SymbolicShape::from_concrete(layout.shape()),
            strides: SmallVec::from_slice(layout.strides()),
            offset: layout.offset(),
        }
    }

    /// Returns a reference to the symbolic shape
    #[inline]
    pub fn shape(&self) -> &SymbolicShape {
        &self.shape
    }

    /// Returns the strides as a slice
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the offset into the underlying storage
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the number of dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Returns true if the shape is fully concrete
    #[inline]
    pub fn is_fully_concrete(&self) -> bool {
        self.shape.is_fully_concrete()
    }

    /// Returns true if any dimension is dynamic
    #[inline]
    pub fn is_dynamic(&self) -> bool {
        self.shape.is_dynamic()
    }

    /// Returns the exact size if all dimensions are concrete
    pub fn size(&self) -> Option<usize> {
        self.shape.size()
    }

    /// Returns the maximum allocation size
    /// Uses max_bound for dynamic dimensions
    pub fn max_size(&self) -> Option<usize> {
        self.shape.max_size()
    }

    /// Returns the required buffer size using max bounds
    pub fn buffer_size(&self) -> Option<usize> {
        self.max_size().map(|s| self.offset + s)
    }

    /// Converts to a concrete Layout if all dimensions are concrete
    pub fn to_concrete(&self) -> Option<Layout> {
        if self.is_fully_concrete() {
            let shape = self.shape.to_concrete()?;
            Some(Layout::new(shape, self.strides.to_vec()))
        } else {
            None
        }
    }

    /// Returns a Layout using max bounds for dynamic dimensions
    /// Returns None if any dynamic dimension lacks a max_bound
    pub fn to_max_layout(&self) -> Option<Layout> {
        let max_shape = self.shape.to_max_shape()?;
        Some(Layout::new(max_shape, self.strides.to_vec()))
    }

    /// Computes contiguous strides using max_bound for dynamic dimensions
    /// Returns None if any dynamic dimension lacks a max_bound
    pub fn compute_strides(shape: &SymbolicShape) -> Option<SmallVec<[usize; 8]>> {
        let ndim = shape.ndim();
        if ndim == 0 {
            return Some(SmallVec::new());
        }

        let mut strides = SmallVec::from_elem(1, ndim);
        for i in (0..ndim - 1).rev() {
            let dim_size = shape.dims()[i + 1].allocation_size()?;
            strides[i] = strides[i + 1] * dim_size;
        }
        Some(strides)
    }

    /// Sets a new offset
    pub fn set_offset(&mut self, offset: usize) {
        self.offset = offset;
    }

    /// Creates a new symbolic layout with a specific dimension
    pub fn with_dim(&self, index: usize, dim: Dim) -> Option<Self> {
        if index >= self.ndim() {
            return None;
        }

        let mut new_dims: Vec<Dim> = self.shape.dims().to_vec();
        new_dims[index] = dim;
        let new_shape = SymbolicShape::new(new_dims);

        // Recompute strides based on new shape
        let strides = Self::compute_strides(&new_shape)?;

        Some(Self {
            shape: new_shape,
            strides,
            offset: self.offset,
        })
    }
}

impl From<&Layout> for SymbolicLayout {
    fn from(layout: &Layout) -> Self {
        Self::from_concrete(layout)
    }
}

impl From<Layout> for SymbolicLayout {
    fn from(layout: Layout) -> Self {
        Self::from_concrete(&layout)
    }
}

impl fmt::Debug for SymbolicLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SymbolicLayout[shape: {:?}, strides: [", self.shape)?;
        for (i, &stride) in self.strides.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", stride)?;
        }
        write!(f, "], offset: {}]", self.offset)
    }
}

impl fmt::Display for SymbolicLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shape: {}, strides: [", self.shape)?;
        for (i, &stride) in self.strides.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", stride)?;
        }
        write!(f, "], offset: {}", self.offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::shape::Shape;

    #[test]
    fn test_from_concrete_layout() {
        let shape = Shape::from(vec![2, 3, 4]);
        let layout = Layout::from_shape(&shape);
        let symbolic = SymbolicLayout::from_concrete(&layout);

        assert!(symbolic.is_fully_concrete());
        assert!(!symbolic.is_dynamic());
        assert_eq!(symbolic.ndim(), 3);
        assert_eq!(symbolic.strides(), &[12, 4, 1]);
        assert_eq!(symbolic.size(), Some(24));
    }

    #[test]
    fn test_with_dynamic_dim() {
        // Shape: [Dynamic(max=100), 3]
        let dims = vec![Dim::dynamic(Some(100)), Dim::Concrete(3)];
        let shape = SymbolicShape::new(dims);
        let symbolic = SymbolicLayout::from_shape(&shape).unwrap();

        assert!(symbolic.is_dynamic());
        assert_eq!(symbolic.ndim(), 2);
        assert_eq!(symbolic.strides(), &[3, 1]); // Strides based on max_bound
        assert_eq!(symbolic.size(), None); // Exact size unknown
        assert_eq!(symbolic.max_size(), Some(300)); // Max size known
        assert!(symbolic.to_concrete().is_none());

        let max_layout = symbolic.to_max_layout().unwrap();
        assert_eq!(max_layout.shape().dims(), &[100, 3]);
    }

    #[test]
    fn test_unbounded_dynamic() {
        // Shape with unbounded dynamic dimension
        let dims = vec![Dim::dynamic(None), Dim::Concrete(3)];
        let shape = SymbolicShape::new(dims);

        // Cannot create layout without max_bound
        assert!(SymbolicLayout::from_shape(&shape).is_none());
    }

    #[test]
    fn test_nonzero_output_shape() {
        // Typical nonzero output: [N, ndim] where N is dynamic
        let ndim = 3usize;
        let max_elements = 1000usize;

        let dims = vec![Dim::dynamic(Some(max_elements)), Dim::Concrete(ndim)];
        let shape = SymbolicShape::new(dims);
        let layout = SymbolicLayout::from_shape(&shape).unwrap();

        assert!(layout.is_dynamic());
        assert_eq!(layout.strides(), &[ndim, 1]); // Strides computed with max_bound
        assert_eq!(layout.max_size(), Some(max_elements * ndim));
    }

    #[test]
    fn test_with_dim() {
        let dims = vec![Dim::Concrete(10), Dim::Concrete(3)];
        let shape = SymbolicShape::new(dims);
        let layout = SymbolicLayout::from_shape(&shape).unwrap();

        // Replace first dim with dynamic
        let new_layout = layout.with_dim(0, Dim::dynamic(Some(50))).unwrap();
        assert!(new_layout.is_dynamic());
        assert_eq!(new_layout.strides(), &[3, 1]);
    }
}
