#!/bin/bash

# Step 1: Install pip if it's not installed
if ! command -v pip &> /dev/null
then
    echo "pip not found, installing..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    rm get-pip.py
fi

# Step 2: Install virtualenv
pip install virtualenv

# Step 3: Create and activate a virtual environment
virtualenv myenv
source myenv/bin/activate

# Step 4: Install required Python packages
pip install transformers
pip install Flask

# Step 5: Train the model (Skip this step if the model is already trained)
python train_model.py

# Step 6: Run the Flask server
cd server
pip install -r requirements.txt
python app.py
