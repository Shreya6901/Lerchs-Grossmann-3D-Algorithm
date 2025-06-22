
# ⛏️ Lerchs-Grossmann 3D Pit Optimization

This project implements the **Lerchs-Grossmann (LG) Algorithm** in Python to optimize open-pit mine design. It helps determine the **ultimate pit limit** that maximizes economic value, while respecting geotechnical slope constraints.



## 📘 About the Algorithm

The **Lerchs-Grossmann algorithm** (1965) is a graph-theoretic solution to the **maximum closure problem** and is widely used in the mining industry to:

- Identify which blocks in a 3D geological model should be extracted.
- Maximize the **net present value (NPV)** of an open-pit mine.
- Ensure stability by enforcing **precedence constraints** based on slope angles.

This Python implementation uses a max-flow based approach to solve the closure problem efficiently and includes slope handling, block value modeling, and 3D visualization.



## 💡 Key Features

- 3D grid block model (voxel-based)
- Random or CSV-based block generation
- Slope constraint enforcement
- Max-flow-based pit optimization
- 3D voxel visualization (pit shell)
- CSV output of extracted blocks


## 📁 File Structure

```
Lerchs-Grossmann-3D-Algorithm
├── lerchs_grossmann.py            # Main Python script
├── mine_pit_dataset.csv           # Dataset 
├── pit_optimization_results.csv   # Output file with extracted blocks
├── README.md                      
```


## 🚀 How to Run

### 🔧 1. Setup Environment

Make sure you have Python 3.7+ installed. Then install required packages:

```
pip install -r requirements.txt
```

### 📦 2. Run the Script

In VS Code or terminal:

```bash
python lerchs_grossmann.py
```

### 🧾 Input Format (`mine_pit_dataset.csv`)

```csv
X,Y,Z,Value,Grade,Rock_Type
0,0,0,100.0,1.5,ore
0,0,1,-8.0,0.0,waste
...
```

Each row defines a block at coordinates `(X, Y, Z)` with its value, ore grade, and rock type.


## 📊 Output

* Terminal output with:

  * Total economic value
  * Number of ore & waste blocks extracted
  * Extraction ratio
* `pit_optimization_results.csv` containing the results
* 3D voxel plot of the pit with:

  * Gold = Extracted Ore
  * Brown = Extracted Waste
  * Gray/Yellow = Unextracted blocks (if shown)


## 🛠️ Modify for Your Dataset

To use your own dataset:

1. Prepare a CSV file in the same format.
2. Replace `mine_pit_dataset.csv` with your file name.
3. Run the script again.


## 📚 References

* Lerchs, H., & Grossmann, I. F. (1965). *Optimum Design of Open-Pit Mines*.
* Muir, D. C. W. (2022). *Lerchs-Grossmann Pit Design: Fifty+ Year History and Code*.

