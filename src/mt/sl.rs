pub struct Solution {
    special: Vec<f64>,
    freedom: Option<Vec<Vec<f64>>>,
}
impl Solution {
    pub fn new(special: Vec<f64>, freedom: Option<Vec<Vec<f64>>>) -> Self {
        Solution { special, freedom }
    }
    
    pub fn special(&self) -> &Vec<f64> {
        &self.special
    }

    pub fn freedom(&self) -> Option<&Vec<Vec<f64>>> {
        match self.freedom {
            Some(ref freedom) => Some(freedom),
            _ => None,
        }
    }
}
