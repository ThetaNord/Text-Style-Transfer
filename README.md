Text Style Transfer

This solution is intended to, as the title implies, tranfer the style of one text to another.
To run the solution, please first make sure that you have all the required Python libraries installed.
You can use "pip install -r requirements.txt" to install everything that is needed, though note that this includes the tensorflow-gpu library.
If you want to use Tensorflow with the CPU only, configure that separately.

To provide material for the program to use, put a text file called "corpus.txt" into the path "models/model-name/" where model-name can be of your own choosing.
You can then run the program with a command of the form:

python -u text-style-transfer.py style-transfer input-file input-model-name output-model-name

The program will create any necessary files for each model in the model's folder automatically.
On subsequent runs it will use existing files and models, so if you want to retrain models, you will have to delete them from the model's folder before running the program.