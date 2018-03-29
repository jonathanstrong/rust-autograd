# Notes from first time using `autograd`

- which form of `ndarray::ArrayBase` to use is very difficult to grok. some kind of documentation explaining which circumstances require which form (and why) would be extremely helpful.
- on that front, refactoring `eval` and `run` to accept a more generic form would offer a lot of flexibility. in my case, I wanted to pass an array view, since my raw data was a fixed-size array, but couldn't.
- in my own use, I swapped `g::<fn>` for `ag::<fn>`. `ag` reminds me of agriculture for whatever reason. I'm a former theano user and I always used `import theano.tensor as T`, so am partial to a single letter.
- I happened across this section of the code:
  
  ```rust
  impl PersistentArray
  {
      pub fn get_as_variable(&self) -> &NdArray
      {
          match *self {
              PersistentArray::Variable(ref a) => a,
              PersistentArray::Constant(_) => panic!("Can't mutate constant tensor"),
          }
      }
  
      #[allow(mutable_transmutes)]
      pub unsafe fn get_as_variable_mut(&self) -> &mut NdArray
      {
          mem::transmute(self.get_as_variable())
      }
      // ...
  }
  ```
  
  This strikes me as fairly dangerous/unexpected by rust standards. Not the `transmute` as much as the `panic!` when the wrong enum variant is used. The quick fix is change the signature to `Option<&NdArray>`. The more fundamental fix is to bridge `Variable` and `Constant` with a trait rather than an enum. I've found myself "trapped" by enums like this many times, and generally structs with traits are a better solution.
- speaking of which, accessing the backing array for a `Tensor` is quite a journey.
- it would be a lot easier to track what code an op runs if the implementation mods weren't private (and thus hidden in docs).
- How can the inner struct of `Tensor` be a `Rc<TensorCore>` (an immutable type) when the point of variables is to mutate their values? Working around the immutable restriction of `Rc` is a big liability as it invalidates the compile-time checks that give rust its safety. See the code I wrote in `Tensor::get_array_mut` that relies/checks on the inner-member having not been copied to return a `&mut`. Alternatives include wrapping the array (or perhaps the entire computational graph, so only one lock) in a mutex, or storing an index in a `Tensor` that can be used to access the array in some global state object, or `Rc<RefCell<TensorCore>>`. Or, perhaps `Tensor` doesn't need to be cheaply cloneable?
