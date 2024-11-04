<h1>U-Net Fine-tuning for Facial Parts Segmentation using LaPa Dataset</h1>
<p>
        This repository provides scripts to fine-tune a U-Net model for segmenting facial parts using the LaPa (Labeled Parts in the Wild) dataset. The model identifies facial regions like eyes, nose, mouth, and hair, making it suitable for applications in facial analysis and augmented reality.
</p>

<h2>Project Structure</h2>

<code>
    .
    ├── main.py                  # Main script to train and validate the model
    ├── model.py                 # Defines the U-Net model architecture
    ├── train.py                 # Training functions, including the optimizer and data loader
    ├── utils.py                 # Helper functions for data preprocessing and transformations
    ├── requirements.txt         # Required packages for this project
</code>

<h2>File Descriptions</h2>
<ul>
        <li><code>main.py</code>: Main script for initializing the training process. Loads data, sets up the model, and manages the overall workflow for training and evaluation.</li>
        <li><code>config.py</code>: Modify the configuration for finetuning based on number of facial parts to be segmented.</li>
        <li><code>data_loader.py</code>: Load the LaPa Dataset based on chosen limit.</li>
        <li><code>unet_model.py</code>: Contains the U-Net model architecture tailored for fine-tuning on the LaPa dataset. Configurable to handle different numbers of classes based on dataset needs.</li>
        <li><code>train.py</code>: Contains functions for training the model, managing epochs, loss calculation, and other necessary training processes.</li>
        <li><code>utils.py</code>: Utility functions for handling data loading and transformations, including preprocessing steps specific to the LaPa dataset.</li>
        <li><code>predict.py</code>: Predict mask using the fine-tuned model.</li>
        <li><code>requirements.txt</code>: Lists all the dependencies needed to set up and run the project. Install these packages using <code>pip install -r requirements.txt</code>.</li>
</ul>

<h2>Installation</h2>

<code>git clone https://github.com/yourusername/LaPa-UNET.git
    cd unet-lapa-finetuning
    pip install -r requirements.txt
</code>

<h2>Usage</h2>
    <p>
        To begin training the model, simply run <code>main.py</code>. This script will load the data, initialize the model, and start the training process:
    </p>
<code>python main.py</code>

<h2>Requirements</h2>
    <p>
        All required packages are listed in <code>requirements.txt</code>. Install them using:
    </p>
<code>pip install -r requirements.txt</code>

<h2>Data</h2>
    <p>
        The model is fine-tuned using the <a href="https://github.com/JDAI-CV/lapa-dataset" target="_blank">LaPa dataset</a>, which contains labeled facial parts. Download the dataset and configure paths within <code>main.py</code> or <code>train.py</code> as needed.
    </p>
