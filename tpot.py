# Function to run TPOT on uploaded CSV data
def run_tpot_automl(csv_file, target_column, problem_type="Classification", generations=5, population_size=20, cv=5):
    """
    Runs TPOT AutoML on uploaded CSV data.

    Args:
        csv_file: File path to the uploaded CSV.
        target_column: The name of the target column.
        problem_type: 'Classification' or 'Regression'.
        generations: Number of generations for TPOT.
        population_size: Population size for TPOT.
        cv: Cross-validation folds.

    Returns:
        A tuple containing:
        - str: Evaluation metric score of the best pipeline.
        - str: Python code for the exported best pipeline.
        - str: Console output/logs from TPOT.
    """
    if csv_file is None:
        return "Error: No CSV file uploaded.", "", ""

    # Explicitly try to shut down any existing Dask clients and workers
    from distributed import Client, get_client
    try:
        client = get_client()
        client.close()
        print("Closed existing Dask client.")
    except ValueError:
        print("No active Dask client found.")
    except Exception as e:
        print(f"Could not close Dask client: {e}")

    # Initialize variables for stdout redirection outside the try block
    old_stdout = None
    text_output = None

    try:
        # Read the CSV file using pandas
        # Use io.StringIO to read the file content as a string if needed,
        # but gr.File often provides a path or a file-like object directly.
        # Let's try reading directly first. If that fails, we can revisit.
        # Removing .decode('utf-8') as it's not needed for the object type
        df = pd.read_csv(csv_file)


        if target_column not in df.columns:
            return f"Error: Target column '{target_column}' not found in the CSV.", "", ""

        # Separate features (X) and target (y)
        y = df[target_column]
        X = df.drop(columns=[target_column])

        # Basic check for non-numeric columns (TPOT requires numeric)
        # This is a simple check, more robust handling might be needed
        non_numeric_cols = X.select_dtypes(exclude=['number']).columns
        if len(non_numeric_cols) > 0:
             return f"Error: Non-numeric columns found: {list(non_numeric_cols)}. TPOT requires numeric data. Please preprocess your data.", "", ""

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        # Initialize and configure TPOT
        if problem_type == "Classification":
            tpot = TPOTClassifier(
                generations=generations,
                population_size=population_size,
                cv=cv,
                random_state=42,
                # verbosity=2, # Removed as it caused an error
                n_jobs=-1 # Use all available cores
            )
            metric = 'balanced_accuracy' # Default classification metric
        elif problem_type == "Regression":
             tpot = TPOTRegressor(
                generations=generations,
                population_size=population_size,
                cv=cv,
                random_state=42,
                # verbosity=2, # Removed as it caused an error
                n_jobs=-1 # Use all available cores
            )
             metric = 'neg_mean_squared_error' # Default regression metric
        else:
             return "Error: Invalid problem type specified. Choose 'Classification' or 'Regression'.", "", ""

        # Redirect TPOT console output to a buffer
        import sys
        old_stdout = sys.stdout
        sys.stdout = text_output = io.StringIO()

        # Fit TPOT on the training data
        tpot.fit(X_train, y_train)

        # Restore stdout
        sys.stdout = old_stdout
        console_output = text_output.getvalue()

        # Evaluate the best pipeline on the test data
        score = tpot.score(X_test, y_test)

        # Export the best pipeline code
        # Use a temporary file name and then read it
        temp_pipeline_file = "tpot_exported_pipeline.py"
        tpot.export(temp_pipeline_file)
        with open(temp_pipeline_file, 'r') as f:
            pipeline_code = f.read()

        # Clean up the temporary file
        import os
        os.remove(temp_pipeline_file)

        return f"Best pipeline test score ({metric}): {score:.4f}", pipeline_code, console_output

    except Exception as e:
        # Restore stdout in case of error during TPOT fit
        if old_stdout is not None:
             sys.stdout = old_stdout
        return f"An error occurred: {e}", "", ""

# Create Gradio Interface
# Use gr.File for uploading the file content directly
# Use gr.Textbox for target column name
# Use gr.Dropdown for problem type
# Use gr.Number for TPOT parameters
# Use gr.Textbox (or gr.Label) for the score output
# Use gr.Code for the pipeline code output
# Use gr.Textbox for console output

interface = gr.Interface(
    fn=run_tpot_automl,
    inputs=[
        gr.File(label="Upload CSV File"),
        gr.Textbox(label="Target Column Name"),
        gr.Dropdown(["Classification", "Regression"], label="Problem Type", value="Classification"),
        gr.Slider(minimum=1, maximum=100, value=5, label="TPOT Generations"),
        gr.Slider(minimum=10, maximum=200, value=20, label="TPOT Population Size"),
        gr.Slider(minimum=2, maximum=10, value=5, label="Cross-Validation Folds (CV)"),
    ],
    outputs=[
        gr.Textbox(label="Best Pipeline Test Score"),
        gr.Code(label="Exported TPOT Pipeline Code", language="python"),
        gr.Textbox(label="TPOT Console Output")
    ],
    title="TPOT-Based AutoML System",
    description="Upload a CSV file, specify the target column and problem type (Classification/Regression), and run TPOT to find the best machine learning pipeline."
)

# Launch the Gradio interface
# In Google Colab, this will provide a public URL
interface.launch(debug=True)
