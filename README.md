# IMDb Score Predictor

### Included Packages:
- tensorflow
- pandas
- scikit-learn (not sklearn)
- matplotlib
- customtkinter

### Steps to Run:
1. Ensure Python 3.11-3.12 and listed packages are installed on your system. (PLEASE NOTE: TensorFlow is not yet compatible with 3.13. You will need to create a venv with a compatible version in order to run)
2. In terminal, move to project directory and run "python main.py"
``python search.py``
3. App will open - decided to train model on each execution as opposed to pre-training, so you will see a "Training Model" prompt before the main menu is available.
   - Please wait, even if window states "Not responding", for model training. Speed depends on user system.

### Package Install
1. Run the following command in terminal:
`` pip install tensorflow pandas scikit-learn matplotlib customtkinter``


### About
This is a machine learning application that predicts IMDb scores based on provided movie attributes (budget, genre, runtime, and so on) via an interactive desktop GUI. Users can either input their own movie information, or select to have one randomly chosen and auto-filled from a provided testset.


### App Preview
<img width="377" alt="PY_Home" src="https://github.com/user-attachments/assets/c39dd485-956d-4868-bdba-66381f3705ea">\
<img width="624" alt="PY_Predict-with-Random" src="https://github.com/user-attachments/assets/bdc13cbb-c2a5-4c23-a88c-708103e30f83">\
<img width="377" alt="PY_Stats" src="https://github.com/user-attachments/assets/da5d9585-2811-40d0-ae05-22e601b28348">
<img width="415" alt="PY_Graphs" src="https://github.com/user-attachments/assets/fd6fe19b-a8fe-4e37-b62d-7ac7ccb5f5a2">


### Future Updates
- Import a movie from IMDb website URL
- Web app front-end (alt)
- Negative values for input (gross profit - bug fix)