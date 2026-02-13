use super::dim::DynamicDimId;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// Thread-local registry for resolved dynamic dimensions
    static RESOLVED_DIMS: RefCell<HashMap<u32, usize>> = RefCell::new(HashMap::new());
}

/// Resolves a dynamic dimension ID to a concrete value
///
/// This is called during snapshot execution when the actual
/// output size becomes known.
pub fn resolve_dimension(id: DynamicDimId, value: usize) {
    RESOLVED_DIMS.with(|dims| {
        dims.borrow_mut().insert(id.id(), value);
    });
}

/// Gets the resolved value for a dynamic dimension ID
///
/// Returns None if the dimension has not been resolved yet.
pub fn get_resolved_dimension(id: DynamicDimId) -> Option<usize> {
    RESOLVED_DIMS.with(|dims| dims.borrow().get(&id.id()).copied())
}

/// Clears all resolved dimensions
///
/// Should be called at the start of each snapshot execution
/// to ensure a clean state.
pub fn clear_resolved_dimensions() {
    RESOLVED_DIMS.with(|dims| {
        dims.borrow_mut().clear();
    });
}

/// Returns the number of currently resolved dimensions
#[cfg(test)]
#[allow(dead_code)]
pub fn resolved_count() -> usize {
    RESOLVED_DIMS.with(|dims| dims.borrow().len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_and_get() {
        clear_resolved_dimensions();

        let id1 = DynamicDimId::new();
        let id2 = DynamicDimId::new();

        // Initially unresolved
        assert!(get_resolved_dimension(id1).is_none());
        assert!(get_resolved_dimension(id2).is_none());

        // Resolve dimension 1
        resolve_dimension(id1, 42);
        assert_eq!(get_resolved_dimension(id1), Some(42));
        assert!(get_resolved_dimension(id2).is_none());

        // Resolve dimension 2
        resolve_dimension(id2, 100);
        assert_eq!(get_resolved_dimension(id1), Some(42));
        assert_eq!(get_resolved_dimension(id2), Some(100));

        // Clear all
        clear_resolved_dimensions();
        assert!(get_resolved_dimension(id1).is_none());
        assert!(get_resolved_dimension(id2).is_none());
    }

    #[test]
    fn test_override_resolution() {
        clear_resolved_dimensions();

        let id = DynamicDimId::new();

        resolve_dimension(id, 10);
        assert_eq!(get_resolved_dimension(id), Some(10));

        // Override with new value
        resolve_dimension(id, 20);
        assert_eq!(get_resolved_dimension(id), Some(20));

        clear_resolved_dimensions();
    }
}
