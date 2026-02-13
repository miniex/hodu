use super::dim::{Dim, DynamicDimId};
use super::shape::Shape;
use smallvec::SmallVec;
use std::fmt;

/// Shape that can contain dynamic dimensions
///
/// This is used during graph capture to represent shapes with data-dependent
/// dimensions (e.g., nonzero output shape depends on input data).
#[derive(Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SymbolicShape {
    dims: SmallVec<[Dim; 8]>,
}

impl SymbolicShape {
    /// Creates a new symbolic shape from a vector of dimensions
    pub fn new(dims: Vec<Dim>) -> Self {
        Self {
            dims: SmallVec::from_vec(dims),
        }
    }

    /// Creates a symbolic shape from a concrete Shape
    pub fn from_concrete(shape: &Shape) -> Self {
        Self {
            dims: shape.dims().iter().map(|&d| Dim::Concrete(d)).collect(),
        }
    }

    /// Creates a symbolic shape with all concrete dimensions
    pub fn concrete(dims: &[usize]) -> Self {
        Self {
            dims: dims.iter().map(|&d| Dim::Concrete(d)).collect(),
        }
    }

    /// Returns the dimensions as a slice
    #[inline]
    pub fn dims(&self) -> &[Dim] {
        &self.dims
    }

    /// Returns the number of dimensions (rank)
    #[inline]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns true if all dimensions are concrete
    #[inline]
    pub fn is_fully_concrete(&self) -> bool {
        self.dims.iter().all(|d| d.is_concrete())
    }

    /// Returns true if any dimension is dynamic
    #[inline]
    pub fn is_dynamic(&self) -> bool {
        self.dims.iter().any(|d| d.is_dynamic())
    }

    /// Converts to a concrete Shape if all dimensions are concrete
    pub fn to_concrete(&self) -> Option<Shape> {
        if self.is_fully_concrete() {
            let concrete: Vec<usize> = self.dims.iter().filter_map(|d| d.concrete_size()).collect();
            Some(Shape::from(concrete))
        } else {
            None
        }
    }

    /// Returns the total size if all dimensions are concrete
    pub fn size(&self) -> Option<usize> {
        if self.is_fully_concrete() {
            Some(self.dims.iter().filter_map(|d| d.concrete_size()).product())
        } else {
            None
        }
    }

    /// Returns the maximum allocation size
    /// Returns None if any dynamic dimension lacks a max bound
    pub fn max_size(&self) -> Option<usize> {
        self.dims
            .iter()
            .map(|d| d.allocation_size())
            .try_fold(1usize, |acc, size| size.map(|s| acc * s))
    }

    /// Returns a Shape using max bounds for dynamic dimensions
    /// Returns None if any dynamic dimension lacks a max bound
    pub fn to_max_shape(&self) -> Option<Shape> {
        let max_dims: Option<Vec<usize>> = self.dims.iter().map(|d| d.allocation_size()).collect();
        max_dims.map(Shape::from)
    }

    /// Returns all dynamic dimension IDs in this shape
    pub fn dynamic_dim_ids(&self) -> Vec<DynamicDimId> {
        self.dims.iter().filter_map(|d| d.dynamic_id()).collect()
    }

    /// Returns the dimension at the given index
    pub fn dim(&self, index: usize) -> Option<&Dim> {
        self.dims.get(index)
    }

    /// Returns the size of a specific dimension if it's concrete
    pub fn dim_size(&self, index: i32) -> Option<usize> {
        let index = if index < 0 {
            (self.ndim() as i32 + index) as usize
        } else {
            index as usize
        };
        self.dims.get(index).and_then(|d| d.concrete_size())
    }
}

impl From<&Shape> for SymbolicShape {
    fn from(shape: &Shape) -> Self {
        Self::from_concrete(shape)
    }
}

impl From<Shape> for SymbolicShape {
    fn from(shape: Shape) -> Self {
        Self::from_concrete(&shape)
    }
}

impl From<Vec<Dim>> for SymbolicShape {
    fn from(dims: Vec<Dim>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for SymbolicShape {
    fn from(dims: &[usize]) -> Self {
        Self::concrete(dims)
    }
}

impl fmt::Debug for SymbolicShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SymbolicShape[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", dim)?;
        }
        write!(f, "]")
    }
}

impl fmt::Display for SymbolicShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_concrete() {
        let shape = Shape::from(vec![2, 3, 4]);
        let symbolic = SymbolicShape::from_concrete(&shape);

        assert!(symbolic.is_fully_concrete());
        assert!(!symbolic.is_dynamic());
        assert_eq!(symbolic.ndim(), 3);
        assert_eq!(symbolic.size(), Some(24));
        assert_eq!(symbolic.to_concrete(), Some(shape));
    }

    #[test]
    fn test_with_dynamic() {
        let dims = vec![Dim::dynamic(Some(100)), Dim::Concrete(3)];
        let symbolic = SymbolicShape::new(dims);

        assert!(!symbolic.is_fully_concrete());
        assert!(symbolic.is_dynamic());
        assert_eq!(symbolic.ndim(), 2);
        assert_eq!(symbolic.size(), None); // Can't compute exact size
        assert_eq!(symbolic.max_size(), Some(300)); // Can compute max size
        assert!(symbolic.to_concrete().is_none());

        let dynamic_ids = symbolic.dynamic_dim_ids();
        assert_eq!(dynamic_ids.len(), 1);
    }

    #[test]
    fn test_unbounded_dynamic() {
        let dims = vec![Dim::dynamic(None), Dim::Concrete(3)];
        let symbolic = SymbolicShape::new(dims);

        assert!(symbolic.is_dynamic());
        assert_eq!(symbolic.max_size(), None); // No max bound
        assert!(symbolic.to_max_shape().is_none());
    }

    #[test]
    fn test_to_max_shape() {
        let dims = vec![Dim::dynamic(Some(100)), Dim::Concrete(3)];
        let symbolic = SymbolicShape::new(dims);

        let max_shape = symbolic.to_max_shape().unwrap();
        assert_eq!(max_shape.dims(), &[100, 3]);
    }
}
