//! Rayon implementations.
use std::ops::{Deref, DerefMut};

use rayon::prelude::*;

use crate::{Gen, View, ViewMut};

impl<'a, S: Send + Sync + 'static, G: Gen<S>> IntoParallelIterator for &'a View<S, G>
where
    &'a S: IntoParallelIterator,
{
    type Iter = <&'a S as IntoParallelIterator>::Iter;

    type Item = <&'a S as IntoParallelIterator>::Item;

    fn into_par_iter(self) -> Self::Iter {
        self.deref().into_par_iter()
    }
}

impl<'a, T: Send + Sync + 'static, G: Gen<T>> IntoParallelIterator for &'a ViewMut<T, G>
where
    &'a T: IntoParallelIterator,
{
    type Item = <&'a T as IntoParallelIterator>::Item;

    type Iter = <&'a T as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        self.deref().into_par_iter()
    }
}

impl<'a, T: Send + Sync + 'static, G: Gen<T>> IntoIterator for &'a mut ViewMut<T, G>
where
    &'a mut T: IntoIterator,
{
    type Item = <<&'a mut T as IntoIterator>::IntoIter as Iterator>::Item;

    type IntoIter = <&'a mut T as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.deref_mut().into_iter()
    }
}

impl<'a, T: Send + Sync + 'static, G: Gen<T>> IntoParallelIterator for &'a mut ViewMut<T, G>
where
    &'a mut T: IntoParallelIterator,
{
    type Item = <&'a mut T as IntoParallelIterator>::Item;

    type Iter = <&'a mut T as IntoParallelIterator>::Iter;

    fn into_par_iter(self) -> Self::Iter {
        self.deref_mut().into_par_iter()
    }
}
