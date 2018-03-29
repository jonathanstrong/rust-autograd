extern crate autograd as g;
//#[macro_use(s)]
extern crate ndarray;

use g::gradient_descent_ops::Optimizer;
use g::Tensor;

const E: f32 = 2.7182817;

fn main() {
    let ref x = g::placeholder(&[-1, 32]);
    let ref y = g::placeholder(&[-1, 3]);

    let ref w0 = g::variable(g::ndarray_ext::glorot_uniform(&[32, 32]));
    let ref b0 = g::variable(g::ndarray_ext::ones(&[1, 32]));
    let ref h0 = g::matmul(x, w0) + b0;
    let ref z0 = g::relu(h0);

    let ref w1 = g::variable(g::ndarray_ext::glorot_uniform(&[32, 3]));
    let ref b1 = g::variable(g::ndarray_ext::ones(&[1, 3]));
    let ref h1 = g::matmul(z0, w1) + b1;
    let ref z1 = g::sigmoid(h1);
    
    fn loss(t: &Tensor, o: &Tensor) -> Tensor {
        let t_ones = g::ones(&t.shape());
        let o_ones = g::ones(&o.shape());
        -1.0 * (t * g::log(o, E) + &g::sub_inplace(t_ones, t) * g::log(&g::sub_inplace(o_ones, o), E))
    }

    let ref mean_loss = g::reduce_mean(&loss(y, z1), &[0, 1], false);
    let ref params = [w0, b0, w1, b1];
    let ref grads = g::grad(&[mean_loss], params);
    let mut adam = g::gradient_descent_ops::Adam::default();
    let ref updates = adam.compute_updates(params, grads);

    let xi = arr(vec![0.0; 32], (1, 32));
    let yi = arr(vec![0.0, 0.5, 1.0], (1, 3));

    fn arr(xs: Vec<f32>, shape: (usize, usize)) -> ndarray::Array<f32, ndarray::IxDyn> {
        ndarray::Array::from_vec(xs).into_shape(shape).unwrap().into_dyn()
    }

    //g::run(updates, &[(x, &ndarray::arr1(&xi[..]).into_dyn()), (y, &ndarray::arr1(&yi[..]).into_dyn())]);
    //let res = g::eval(updates, &[(x, &xi), (y, &yi)]);
    g::run(updates, &[(x, &xi), (y, &yi)]);
    //println!("{:?}", ndarray::aview1(&xi).into_shape((1, 32)).unwrap().into_dyn());
    //println!("{:?}", view(&xi, (1, 32)));


    //let ref a = g::placeholder(&[-1]);
    //let ref b = g::placeholder(&[-1]);
    //let ref entropy = loss(a, b);
    //let ref sum = a + b;
    ////let res = g::eval(&[sum], &[(a, &[1.0]), (b, &[1.0]));//(b, &ndarray::arr1(&[0.25]))]);
    //let c = ndarray::arr1(&[0.0, 0.0, 0.5, 0.5, 1.0, 1.0]).into_dyn();
    //let d = ndarray::arr1(&[0.01, 0.25, 0.45, 0.55, 0.75, 0.99]).into_dyn();
    //let res = entropy.eval(&[(a, &c), (b, &d)]);
    //println!("{:?}", c);
    //println!("{:?}", d);
    //println!("entropy = {:?}", res);

}


