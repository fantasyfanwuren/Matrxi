mod sl;
pub use self::sl::Solution;

use crate::vq::VectorQuantity;

use std::ops::Add;
use std::ops::Mul;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::{fmt, thread};

#[derive(Debug, Clone, PartialEq)]
pub struct Position {
    row: usize,
    col: usize,
}

impl Position {
    pub fn new(row: usize, col: usize) -> Position {
        Position { row, col }
    }
    pub fn next(&self) -> Self {
        Position {
            row: self.row + 1,
            col: self.col + 1,
        }
    }
}
// 初等行变换
pub trait RowOperation {
    // 加变换
    fn add(&mut self, static_row: usize, target_row: usize, static_num: f64, target_num: f64);
    // 对换变换
    fn transposition(&mut self, row1: usize, row2: usize);
    // 倍乘变换
    fn multiply(&mut self, row: usize, multiply_by: f64, divide_by: f64);
}
/// # Matrix为一个矩阵

#[derive(PartialEq, Debug, Clone)]
pub struct Matrix {
    elements: Vec<Vec<f64>>,
    rref: Option<Vec<Vec<f64>>>,
    pivots: Option<Vec<Position>>,
}

impl Matrix {
    /**#### 获取一个Vec<Vec<A>>参数的所有权,将其转化为一个 Matrix,
     * 首先要求泛型A可以转化为f64,否则编译器将提示错误,
     * 另外如果elements中的各行元素个数若不相等,会在运行中panic,
     * 转化后的Matrix为初始化状态,它的rref(简化阶梯行)和pivots(主元位置集合)都是None.
     */
    pub fn from<A>(elements: Vec<Vec<A>>) -> Self
    where
        f64: From<A>,
    {
        // 首先判断elements中的各个Vec<A>是否元素个数相同
        if elements.len() != 0 {
            let should_len = elements[0].len();
            for item in elements.iter() {
                if should_len != item.len() {
                    panic!("Each row must have the same number of elements!");
                }
            }
        }
        // 将elements 转化为Vec<Vec<f64>>
        let elements: Vec<Vec<f64>> = elements
            .into_iter()
            .map(|x| x.into_iter().map(|x| x.into()).collect::<Vec<f64>>())
            .collect();
        Matrix {
            elements,
            rref: None,
            pivots: None,
        }
    }

    /**#### 在矩阵的最下方添加一行 */
    pub fn push_row<A>(&mut self, row_elements: Vec<A>)
    where
        f64: From<A>,
    {
        // 首先判断row_elements中元素的个数,与self.elements的列数是否相等
        if self.col_count() != row_elements.len() {
            panic!("The number of elements in row_elements should be equal to the number of columns in the matrix.");
        }
        // 进行push操作
        let row_elements = row_elements.into_iter().map(|x| x.into()).collect();
        self.elements.push(row_elements);

        // 重置rref 和 pivots
        self.rref = None;
        self.pivots = None;
    }

    /**#### 返回当前矩阵的行数 */
    pub fn row_count(&self) -> usize {
        self.elements.len()
    }

    /**#### 返回当前矩阵的列数 */
    pub fn col_count(&self) -> usize {
        if self.row_count() == 0 {
            0
        } else {
            self.elements[0].len()
        }
    }

    /**#### 返回矩阵的简易阶梯行的引用,
     * 若矩阵为初始化状态(rref和pivots都是None),则该函数会计算简化阶梯行,因此会改变矩阵的内部数据
     * 但一旦计算过一次后,之后的所有与简化阶梯行rref相关的函数,都不会再进行重复计算
     * 因此一般创建一个矩阵的时候,推荐使用let mut创建
     */
    pub fn rref(&mut self) -> &Vec<Vec<f64>> {
        loop {
            match self.rref {
                Some(ref result) => break result,
                _ => {
                    self.rref = Some(self.elements.clone());
                    let mut current_pivot_position = Position::new(0, 0);
                    let mut pivots: Vec<Position> = vec![];
                    // 变换成阶梯行
                    while let Some(position) = self.get_pivot(&current_pivot_position) {
                        pivots.push(position.clone());
                        self.lay_eggs(&position);
                        current_pivot_position = position.next();
                    }
                    // 将阶梯行变换成简易阶梯行
                    for povot_position in pivots.iter().rev() {
                        self.bubbling(povot_position);
                    }

                    // 更新主元位置
                    self.pivots = Some(pivots)
                }
            }
        }
    }

    /**#### 返回矩阵的主元位置集合 */
    pub fn pivots(&mut self) -> &Vec<Position> {
        loop {
            match self.pivots {
                Some(ref pivots) => break pivots,
                _ => {
                    let _ = self.rref();
                }
            }
        }
    }

    /**#### 判断矩阵是否相容,若相容返回true,若不相容返回false */
    pub fn is_compatibility(&mut self) -> bool {
        let pivots = self.pivots();
        match pivots.last() {
            Some(position) => {
                if position.col == self.col_count() - 1 {
                    false
                } else {
                    true
                }
            }
            _ => true,
        }
    }

    /**#### 判断矩阵是否存在唯一解 */
    pub fn is_unique_solution(&mut self) -> bool {
        if self.is_compatibility() {
            let pivots = self.pivots();
            // 若相容,且主元个数与列数-1相等,存在唯一解
            // 若相容,且主元个数与列数-1不相等,则
            pivots.len() + 1 == self.col_count()
        } else {
            // 若不相容,就是无解
            false
        }
    }

    /**#### 判断两个矩阵是否等价 */
    pub fn is_row_equal(&mut self, other: &mut Self) -> bool {
        let rref = self.rref();
        let other = other.rref();
        *rref == *other
    }

    /**#### 将一个向量添加到矩阵的尾列,
     * 这将获取被添加向量的所有权,因此若希望添加后继续使用该向量,需要clone
     */
    pub fn push_vq(&mut self, vq: VectorQuantity) {
        // 先检测向量的元素个数是否与矩阵的列数相等,若不等则panic.
        assert_eq!(vq.row_count(), self.row_count());
        // 逐行添加
        let mut vq_iter = vq.target().iter();
        for row in self.elements.iter_mut() {
            if let Some(&vq_item) = vq_iter.next() {
                row.push(vq_item);
            }
        }
        self.pivots = None;
        self.rref = None;
    }

    /**#### 判定一个向量vq是否可以由矩阵的各列组合而成 */
    pub fn can_make(&self, vq: &VectorQuantity) -> bool {
        let mut a = self.clone();
        let b = vq.clone();
        a.push_vq(b);
        a.is_compatibility()
    }

    /**#### 判定矩阵可以组合出任意与矩阵行数相同的向量 */
    pub fn can_make_all(&mut self) -> bool {
        // 将矩阵rref化,若已经rref化,则直接获取
        let pivots = self.pivots();
        // 判断矩阵的每一行都有一个主元
        // 因为每一行只能有一个主元,所以若每一行都有主元,则主元的个数等于行数
        pivots.len() == self.row_count()
    }

    /**#### 创建一个单位矩阵 */
    pub fn cell(size: usize) -> Self {
        let mut elements: Vec<Vec<f64>> = Vec::with_capacity(size);
        for current_row in 0..size {
            let mut row_elements: Vec<f64> = Vec::with_capacity(size);
            for current_col in 0..size {
                if current_col == current_row {
                    row_elements.push(1.0);
                } else {
                    row_elements.push(0.0);
                }
            }
            elements.push(row_elements);
        }

        Matrix::from(elements)
    }

    /**#### 判断一个矩阵各列是否线性相关 linearly_independent*/
    pub fn is_linearly_independent(&mut self) -> bool {
        // 将矩阵rref化,若已经rref化,则直接获取
        let pivots = self.pivots();
        // 线性无关的充分必要条件是A*X = 0,只有平凡解
        // 若希望A*X = 0有唯一解,那么就意味着 A的每一列都有主元
        // 因为每列最多有一个主元,则列数等于pivots的个数即可判定是否线性无关
        !(pivots.len() == self.col_count())
    }

    /**#### 若该矩阵为映射T的标准矩阵,判断是否为单射(Injective)*/
    pub fn is_injective(&mut self) -> bool {
        // 将矩阵rref化,若已经rref化,则直接获取
        let pivots = self.pivots();
        // 单设的充分必要条件为A*x = 0 只有平凡解
        // => A*x = 0有唯一解
        // => A的每一列都有主元
        // => 因为同一列最多一个主元,则每一列都有一个主元,则充分必要条件为主元的数量 = 列数
        pivots.len() == self.col_count()
    }

    /**#### 创建一个row*col的零矩阵 */
    pub fn zero(row: usize, col: usize) -> Matrix {
        assert!(row != 0 && col != 0);

        let mut elements = Vec::with_capacity(row);

        for _ in 0..row {
            elements.push(vec![0.0; col]);
        }

        Matrix {
            elements,
            rref: None,
            pivots: None,
        }
    }

    /**#### 多线程计算矩阵与向量相乘
     * 相乘后等式右侧变量都会失去所有权
     */
    pub fn thread_mul_vq(self, vq: VectorQuantity) -> VectorQuantity {
        // 判断二者是否能够线程
        assert_eq!(self.col_count(), vq.row_count());

        let target = Arc::new(Mutex::new(vec![0.0; self.row_count()]));

        // 每行开启一个线程
        let threads: Vec<JoinHandle<()>> = (0..self.row_count())
            .into_iter()
            .map(|row| {
                // 复制一下target指针,准备将所有权添加进新线程
                let target = Arc::clone(&target);

                // 克隆一下对应线程的矩阵行,以备添加到线程中
                let matrix_row = self.elements[row].clone();

                // 复制一下对应线程的向量vec,以备添加到线程中
                let vq_row = vq.target().clone();

                // 采用move转移相关变量的所有权到线程中
                thread::spawn(move || {
                    // 初始化要计算的元素为0
                    let mut vq_element = 0.0;

                    // 计算对应线程的对应位置的向量的单个元素
                    for (num, element) in matrix_row.into_iter().enumerate() {
                        vq_element += element * vq_row[num];
                    }

                    // 将元素添加到对应位置
                    let mut target = target.lock().unwrap();
                    target[row] = vq_element;
                })
            })
            .collect();

        // 等待所有线程结束
        for thread in threads {
            thread.join().unwrap();
        }

        // 提取target锁中的vec
        let target = target.lock().unwrap().to_vec();
        VectorQuantity::new(target)
    }

    /**#### 多线程计算矩阵与矩阵相乘
     * 相乘后等式右侧变量都会失去所有权
     */
    pub fn thread_mul_matrix(self, other: Matrix) -> Matrix {
        // 判断是否允许相乘
        assert_eq!(self.col_count(), other.row_count());

        // 定义一个矩阵的原子引用互斥锁,方便在多线程中修改数据
        let elements = vec![vec![0.0; other.col_count()]; self.row_count()]; //Matrix::zero(self.row_count(), other.col_count());
        let elements = Arc::new(Mutex::new(elements));

        // 对线程进行布置

        let mut threads = Vec::new();
        for row in 0..self.row_count() {
            for col in 0..other.col_count() {
                // 拷贝数据
                let row_elements = self.get_row_clone(row);
                let col_elements = other.get_col_clone(col);
                let elements = Arc::clone(&elements);

                // 创建线程,在线程内部对elements进行改变
                let handle = thread::spawn(move || {
                    let mut element = 0.0;
                    for (num, _) in row_elements.iter().enumerate() {
                        element += row_elements[num] * col_elements[num];
                    }
                    let mut elements = elements.lock().unwrap();
                    elements[row][col] = element;
                });
                threads.push(handle);
            }
        }

        for handle in threads {
            handle.join().unwrap();
        }

        let elements = elements.lock().unwrap().to_owned();
        Matrix::from(elements)
    }

    /*获取一个矩阵中的一行数据 */
    pub fn get_row_clone(&self, row: usize) -> Vec<f64> {
        // 首先判断row的大小是否合规
        assert!(row < self.row_count());
        self.elements[row].clone()
    }

    /*获取一个矩阵中以列的数据 */
    pub fn get_col_clone(&self, col: usize) -> Vec<f64> {
        // 首先判断col的大小是否合规
        assert!(col < self.col_count());
        let mut col_elements = Vec::new();
        for row in 0..self.row_count() {
            col_elements.push(self.elements[row][col]);
        }
        col_elements
    }

    /**多线程运算矩阵与数字相乘 */

    pub fn thread_mul_num<B>(self, num: B) -> Matrix
    where
        B: Into<f64>,
    {
        let row_count = self.row_count();
        let num = num.into();

        // 每行开启一个线程
        let handles: Vec<_> = (0..row_count)
            .into_iter()
            .map(|row| {
                let row_elements = self.get_row_clone(row);
                thread::spawn(move || -> Vec<f64> {
                    row_elements
                        .into_iter()
                        .map(|element| element * num)
                        .collect::<Vec<f64>>()
                })
            })
            .collect();

        // 汇总计算结果
        let mut elements = Vec::new();
        for handle in handles {
            elements.push(handle.join().unwrap());
        }
        Matrix::from(elements)
    }

    /**#### 计算出矩阵的解,返回值为一个Option<Solution>若解不存在,则返回None,若解存在则返回 Solution */
    // pub fn solution(&mut self) -> Option<Solution> {
    //     if true {
    //         None
    //     } else {
    //         let special = vec![0f64, 1f64, 1f64, 3f64];
    //         let freedom = Some(vec![vec![0f64, 1f64, 1f64, 3f64]]);
    //         Some(Solution::new(special, freedom))
    //     }
    // }

    /**#### 以下为私有函数: */
    // 计算一个子矩阵的主元位置,其中row,col代表子矩阵左上角的坐标值
    fn get_pivot(&mut self, position: &Position) -> Option<Position> {
        let row_limit = self.elements.len();
        let col_limit = self.elements[0].len();
        let mut target_position = position.clone();
        if target_position.row == row_limit {
            return None;
        };

        loop {
            if target_position.col == col_limit {
                return None;
            }
            let mut max_row = target_position.row;
            // 将子矩阵中
            let rref = self.rref.as_ref().unwrap();
            for row in target_position.row + 1..row_limit {
                if rref[row][target_position.col].abs() > rref[max_row][target_position.col].abs() {
                    max_row = row;
                }
            }

            self.transposition(target_position.row, max_row);
            let rref = self.rref.as_ref().unwrap();
            if rref[target_position.row][target_position.col] == 0.0 {
                target_position.col += 1;
            } else {
                break;
            }
        }
        Some(target_position)
    }
    // // 通过倍加行变换将主元下方的元素变成0,将其形象地比喻为下蛋
    // // 参数为主元位置
    fn lay_eggs(&mut self, position: &Position) {
        //println!("下蛋{:?}", position)
        let rref = self.rref.as_ref().unwrap();
        let host_element = rref[position.row][position.col];
        for target_row in position.row + 1..self.elements.len() {
            let rref = self.rref.as_ref().unwrap();
            let target_element = rref[target_row][position.col];
            //let num = -(target_element / host_element);
            self.add(position.row, target_row, -target_element, host_element);
        }
    }

    // 把每个主元上方的各元素变成0,并将主元元素变成1,将其形象地比喻为冒泡
    // 参数为主元位置
    fn bubbling(&mut self, position: &Position) {
        //println!("冒泡{:?}", position);
        let rref = self.rref.as_ref().unwrap();
        let host_element = rref[position.row][position.col];
        for target_row in (0..position.row).rev() {
            let rref = self.rref.as_ref().unwrap();
            let target_element = rref[target_row][position.col];
            //let num = -(target_element / host_element);
            self.add(position.row, target_row, -target_element, host_element);
        }
        self.multiply(position.row, 1.0, host_element);
    }
}

impl RowOperation for Matrix {
    fn add(&mut self, static_row: usize, target_row: usize, static_num: f64, target_num: f64) {
        let rref = self.rref.as_mut().unwrap();
        let mut bridge = rref[static_row].clone();
        let mut col = 0;
        bridge = bridge
            .iter()
            .map(|&x| {
                let temp = static_num * x + rref[target_row][col] * target_num;
                col += 1;
                temp
            })
            .collect();
        rref[target_row] = bridge;
    }

    fn multiply(&mut self, row: usize, multiply_by: f64, divide_by: f64) {
        let rref = self.rref.as_mut().unwrap();
        rref[row] = rref[row]
            .iter()
            .map(|x| x * multiply_by / divide_by)
            .collect();
    }

    fn transposition(&mut self, row1: usize, row2: usize) {
        if row1 == row2 {
            return;
        }
        let rref = self.rref.as_mut().unwrap();
        rref.swap(row1, row2);
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut output = String::new();

        // 打印矩阵
        output.push_str("Matrix:\n");
        for row in self.elements.iter() {
            if row.len() == 0 {
                output.push_str("vec[]")
            } else {
                output.push_str(&format!("[{}", row[0]));
                for item in row.iter().skip(1) {
                    output.push_str(&format!(",{}", item));
                }
                output.push_str("]\n");
            }
        }
        output.push_str("\n");

        // 打印简易行列式
        output.push_str("RREF:\n");
        match self.rref {
            None => output.push_str("None"),
            Some(ref rref) => {
                for row in rref.iter() {
                    if row.len() == 0 {
                        output.push_str("vec[]")
                    } else {
                        output.push_str(&format!("[{}", row[0]));
                        for item in row.iter().skip(1) {
                            output.push_str(&format!(",{} ", item));
                        }
                        output.push_str("]\n");
                    }
                }
            }
        }
        output.push_str("\n");

        // 打印主元
        output.push_str("PIVOTS:\n");
        match self.pivots {
            None => output.push_str("None"),
            Some(ref pivots) => {
                if pivots.len() == 0 {
                    output.push_str("vec[]");
                } else {
                    output.push_str(&format!("[{:?}", pivots[0]));
                    for item in pivots.iter().skip(1) {
                        output.push_str(&format!(",{:?}", item));
                    }
                    output.push_str("]");
                }
            }
        }
        output.push_str("\n\n");
        write!(f, "{}", output)
    }
}

/**
 * #### 线性变换:让一个矩阵与一个向量相乘,得到另外一个向量,通常在线性代数中的被描述为:A * x = b,
 * 值得注意的是:相乘后无论是矩阵A,还是rhs向量,都会失去所有权,因此若不希望他们失去所有权,请使用clone
 */
impl Mul<VectorQuantity> for Matrix {
    type Output = VectorQuantity;
    fn mul(self, vector: VectorQuantity) -> Self::Output {
        // 判断是否具有能够相乘的条件
        assert_eq!(self.col_count(), vector.row_count());

        let mut target: Vec<f64> = vec![];

        for row in 0..self.row_count() {
            let mut value = 0.0;
            for col in 0..vector.row_count() {
                value += self.elements[row][col] * vector.target()[col]
            }
            target.push(value);
        }
        VectorQuantity::new(target)
    }
}

/**
 * #### 矩阵与矩阵相乘运算,后期需要优化为复杂度更低的算法
 * 相乘后,两个矩阵都会被drop,若希望继续使用,需要用户提前使用clone
*/
impl Mul<Matrix> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: Matrix) -> Self::Output {
        // 首先判断是否可以相乘
        assert_eq!(self.col_count(), rhs.row_count());

        // 分析:结果的第row行第col列的元素等于,self的第row行与rhs的第col列各个元素乘积和
        let total_row = self.row_count();
        let total_col = rhs.col_count();
        let total_step = self.col_count();

        let mut matrix = Matrix::zero(total_row, total_col);
        for row in 0..total_row {
            for col in 0..total_col {
                for step in 0..total_step {
                    matrix.elements[row][col] += self.elements[row][step] * rhs.elements[step][col];
                }
            }
        }
        matrix
    }
}

/**#### 矩阵与矩阵相加
 * 相加后两个矩阵都会被drop,若希望继续使用,需要用户提前clone
*/
impl Add<Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Self::Output {
        // 判断两个允许相加的条件是否成立
        assert_eq!(self.row_count(), rhs.row_count());
        assert_eq!(self.col_count(), rhs.col_count());

        let row_count = self.row_count();
        let col_count = self.col_count();

        let mut elements = vec![vec![0.0; col_count]; row_count];

        for row in 0..self.row_count() {
            for col in 0..self.col_count() {
                elements[row][col] = self.elements[row][col] + rhs.elements[row][col];
            }
        }
        Matrix::from(elements)
    }
}

/**#### 定义矩阵乘以一个浮点数 */
impl<T: Into<f64>> Mul<T> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        let elements = self
            .elements
            .into_iter()
            .map(|row| row.into_iter().map(|element| element * rhs).collect())
            .collect();
        Matrix::from(elements)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_matrix_from() {
        let a = Matrix::from(vec![vec![1, 2, 3, 4], vec![2, 3, 4, 5]]);
        let b = Matrix {
            elements: vec![vec![1f64, 2f64, 3f64, 4f64], vec![2f64, 3f64, 4f64, 5f64]],
            pivots: None,
            rref: None,
        };
        assert_eq!(a, b);

        let s: Vec<Vec<i32>> = vec![];
        let c = Matrix::from(s);
        let q: Vec<Vec<f64>> = vec![];
        let d = Matrix {
            elements: q,
            rref: None,
            pivots: None,
        };
        assert_eq!(c, d);
    }

    #[test]
    #[should_panic]
    fn test_panic_matrix_from() {
        let a = Matrix::from(vec![vec![2, 3, 4], vec![2, 3, 4, 5]]);
        println!("{a}");
    }

    #[test]
    fn test_matrix_push_row() {
        let mut a = Matrix::from(vec![vec![2, 3, 4, 5], vec![2, 3, 4, 5]]);
        a.push_row(vec![2.0f32, 3.0f32, 4.0f32, 5.0f32]);

        let b = Matrix::from(vec![vec![2, 3, 4, 5], vec![2, 3, 4, 5], vec![2, 3, 4, 5]]);
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic]
    fn test_matrix_push_row_panic() {
        let mut a = Matrix::from(vec![vec![2, 3, 4, 5], vec![2, 3, 4, 5]]);
        a.push_row(vec![2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32]);
    }

    #[test]
    fn test_rref() {
        let mut a = Matrix::from(vec![
            vec![1, -2, 1, 0],
            vec![0, 2, -8, 8],
            vec![5, 0, -5, 10],
        ]);
        let _ = a.rref();

        let b = Matrix {
            elements: vec![
                vec![1f64, -2f64, 1f64, 0f64],
                vec![0f64, 2f64, -8f64, 8f64],
                vec![5f64, 0f64, -5f64, 10f64],
            ],
            rref: Some(vec![
                vec![1f64, 0f64, 0f64, 1f64],
                vec![0f64, 1f64, 0f64, 0f64],
                vec![0f64, 0f64, 1f64, -1f64],
            ]),
            pivots: Some(vec![
                Position { row: 0, col: 0 },
                Position { row: 1, col: 1 },
                Position { row: 2, col: 2 },
            ]),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn test_matrix_mul() {
        let a = Matrix::from(vec![
            vec![1f64, 2f64, 3f64],
            vec![1f64, 2f64, 3f64],
            vec![1f64, 2f64, 3f64],
            vec![1f64, 2f64, 3f64],
        ]);

        let b = VectorQuantity::new(vec![2f64, 3f64, 4f64]);
        let b1 = VectorQuantity::new(vec![20f64, 20f64, 20f64, 20f64]);
        let mul_result = a * b;
        assert_eq!(mul_result, b1);
    }

    #[test]
    fn test_pivots() {
        let mut a = Matrix::from(vec![
            vec![0, 1, -4, 8],
            vec![2, -3, 2, 1],
            vec![4, -8, 12, 1],
        ]);

        let b = vec![
            Position { row: 0, col: 0 },
            Position { row: 1, col: 1 },
            Position { row: 2, col: 3 },
        ];
        assert_eq!(*a.pivots(), b);
    }

    #[test]
    fn test_is_compatibility() {
        // 测试不相容
        let mut a = Matrix::from(vec![
            vec![0, 1, -4, 8],
            vec![2, -3, 2, 1],
            vec![4, -8, 12, 1],
        ]);
        assert!(!a.is_compatibility());
        // 测试相容
        let mut a = Matrix::from(vec![
            vec![1, -2, 1, 0],
            vec![0, 2, -8, 8],
            vec![5, 0, -5, 10],
        ]);
        assert!(a.is_compatibility());

        // 测试0
        let mut a = Matrix::from(vec![vec![0, 0, 0, 0], vec![0, 0, 0, 0], vec![0, 0, 0, 0]]);
        assert!(a.is_compatibility());
        println!("{:?}", a);
    }

    #[test]
    fn is_unique_solution() {
        // 测试不相容
        let mut a = Matrix::from(vec![
            vec![0, 1, -4, 8],
            vec![2, -3, 2, 1],
            vec![4, -8, 12, 1],
        ]);
        assert!(!a.is_unique_solution());
        // 测试相容且解唯一
        let mut a = Matrix::from(vec![
            vec![1, -2, 1, 0],
            vec![0, 2, -8, 8],
            vec![5, 0, -5, 10],
        ]);
        assert!(a.is_unique_solution());

        // 测试相容且解不唯一
        let mut a = Matrix::from(vec![vec![1, 0, -5, 1], vec![0, 1, 1, 4], vec![0, 0, 0, 0]]);
        assert!(!a.is_unique_solution());

        // 测试0矩阵
        let mut a = Matrix::from(vec![vec![0, 0, 0, 0], vec![0, 0, 0, 0], vec![0, 0, 0, 0]]);
        assert!(!a.is_unique_solution());
    }

    #[test]
    fn test_is_row_equal() {
        let mut a = Matrix::from(vec![
            vec![0, 1, -4, 8],
            vec![2, -3, 2, 1],
            vec![4, -8, 12, 1],
        ]);
        let mut b = Matrix::from(vec![
            vec![2, -3, 2, 1],
            vec![0, 1, -4, 8],
            vec![4, -8, 12, 1],
        ]);

        assert_eq!(a.rref(), b.rref());
        assert!(a.is_row_equal(&mut b));
    }

    #[test]
    fn test_push_vq() {
        let mut a = Matrix::from(vec![
            vec![0, 1, -4, 8],
            vec![2, -3, 2, 1],
            vec![4, -8, 12, 1],
        ]);

        let vq = VectorQuantity::new(vec![2.0, 3.0, 4.0]);

        a.push_vq(vq);
        let b = Matrix::from(vec![
            vec![0, 1, -4, 8, 2],
            vec![2, -3, 2, 1, 3],
            vec![4, -8, 12, 1, 4],
        ]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_can_make() {
        // 测试不相容
        let a = Matrix::from(vec![vec![0, 1, -4], vec![2, -3, 2], vec![4, -8, 12]]);

        let vq = VectorQuantity::new(vec![8.0, 1.0, 1.0]);
        assert!(!a.can_make(&vq));

        // 测试相容
        let a = Matrix::from(vec![
            vec![1, -2, 1, 0],
            vec![0, 2, -8, 8],
            vec![5, 0, -5, 10],
        ]);
        let vq = VectorQuantity::new(vec![0.0, 8.0, 10.0]);
        assert!(a.can_make(&vq));
    }

    #[test]
    fn test_can_make_all() {
        // 测试矩阵的列不能生成R(m)
        let mut a = Matrix::from(vec![vec![1, 3, 4], vec![-4, 2, -6], vec![-3, -2, -7]]);
        assert!(!a.can_make_all());
        // 测试矩阵的列可以生成R(m)
        let mut a = Matrix::from(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
        assert!(a.can_make_all());
        // 习题37
        let mut a = Matrix::from(vec![
            vec![-5, 10, -5, 4],
            vec![8, 3, -4, 7],
            vec![4, -9, 5, -3],
            vec![-3, -2, 5, 4],
        ]);
        println!("1-9习题37:{}", a.can_make_all());
        assert_eq!(a.can_make_all(), false);
        //习题39
        let mut a = Matrix::from(vec![
            vec![4, -7, 3, 7, 5],
            vec![6, -8, 5, 12, -8],
            vec![-7, 10, -8, -9, 14],
            vec![3, -5, 4, 2, -6],
            vec![-5, 6, -6, -7, 3],
        ]);
        println!("1-9习题39:{}", a.can_make_all());
        assert_eq!(a.can_make_all(), false);
    }

    #[test]
    fn test_matrix_cell() {
        let a = Matrix::cell(3);
        let b = Matrix::from(vec![vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_() {
        let mut a = Matrix::from(vec![vec![5, 7, 9], vec![0, 2, 4], vec![0, -6, -8]]);
        //println!("{}", a.is_linearly_independent()); // false(线性无关) // 答案:线性无关,正确
        assert_eq!(a.is_linearly_independent(), false);

        // 习题3
        let mut a = Matrix::from(vec![vec![1, -3], vec![-3, 9]]);
        //println!("{}", a.is_linearly_independent()); // true. 答案:线性相关,正确
        assert_eq!(a.is_linearly_independent(), true);

        // 习题5
        let mut a = Matrix::from(vec![
            vec![0, -8, 5],
            vec![3, -7, 4],
            vec![-1, 5, -4],
            vec![1, -3, 2],
        ]);
        //println!("{}", a.is_linearly_independent()); //false 答案:线性无关,正确
        assert_eq!(a.is_linearly_independent(), false);
        // 习题6
        let mut a = Matrix::from(vec![
            vec![1, 4, -3, 0],
            vec![-2, -7, 5, 1],
            vec![-4, -5, 7, 5],
        ]);
        //println!("{}", a.is_linearly_independent()); // true
        assert_eq!(a.is_linearly_independent(), true);

        // 习题20
        let mut a = Matrix::from(vec![vec![1, -2, 0], vec![4, 5, 0], vec![-7, 3, 0]]);
        //println!("{}", a.is_linearly_independent()) //true 正确
        assert_eq!(a.is_linearly_independent(), true);
    }

    #[test]
    fn test_is_injective() {
        let mut a = Matrix::from(vec![vec![3, 1], vec![5, 7], vec![1, 3]]);
        assert_eq!(a.is_injective(), true);
    }

    #[test]
    fn test_zero() {
        let a = Matrix::zero(2, 3);
        let b = Matrix::from(vec![vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0]]);
        let c = Matrix::from(vec![vec![0, 0, 0], vec![0, 0, 0]]);
        assert_eq!(a, b);
        assert!(b == c);
    }

    #[test]
    #[should_panic]
    fn test_zero_panic() {
        let a = Matrix::zero(0, 1);
        println!("{a}");
    }

    #[test]
    fn test_matrix_mul_matrix() {
        let a = Matrix::from(vec![vec![2, 3], vec![1, -5]]);
        let b = Matrix::from(vec![vec![4, 3, 6], vec![1, -2, 3]]);
        let should_result = Matrix::from(vec![vec![11, 0, 21], vec![-1, 13, -9]]);
        assert_eq!(a * b, should_result);
    }

    #[test]
    fn test_thread_mul_vq() {
        let a = Matrix::from(vec![
            vec![1f64, 2f64, 3f64],
            vec![1f64, 2f64, 3f64],
            vec![1f64, 2f64, 3f64],
            vec![1f64, 2f64, 3f64],
        ]);

        let b = VectorQuantity::new(vec![2f64, 3f64, 4f64]);
        let b1 = VectorQuantity::new(vec![20f64, 20f64, 20f64, 20f64]);
        let mul_result = a.thread_mul_vq(b);
        assert_eq!(mul_result, b1);
    }

    #[test]
    fn test_thread_mul_matrix() {
        let a = Matrix::from(vec![vec![2, 3], vec![1, -5]]);
        let b = Matrix::from(vec![vec![4, 3, 6], vec![1, -2, 3]]);
        let should_result = Matrix::from(vec![vec![11, 0, 21], vec![-1, 13, -9]]);
        assert_eq!(a.thread_mul_matrix(b), should_result);
    }

    #[test]
    fn test_matrix_add() {
        let a = Matrix::from(vec![vec![2, 3], vec![1, -5]]);
        let c = Matrix::from(vec![vec![2, 3], vec![1, -5]]);
        let b = Matrix::from(vec![vec![4, 6], vec![2, -10]]);
        assert_eq!(a + c, b);
    }

    #[test]
    fn test_matrix_mul_num() {
        let a = Matrix::from(vec![vec![2, 3], vec![1, -5]]);
        let c = Matrix::from(vec![vec![2, 3], vec![1, -5]]);
        let b = Matrix::from(vec![vec![4, 6], vec![2, -10]]);
        let d = Matrix::from(vec![vec![2, 3], vec![1, -5]]);
        let e = Matrix::from(vec![vec![2, 3], vec![1, -5]]);
        assert_eq!(a * 2, b);
        assert_eq!(c * 2.0, b);
        assert_eq!(d.thread_mul_num(2), b);
        assert_eq!(e.thread_mul_num(2.0), b);
    }
}
