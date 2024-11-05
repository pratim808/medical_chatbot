
# Project Title

End-to-end-Medical-Chatbot-using-Llama3-8b-8192


# End-to-end-Medical-Chatbot-using-Llama2

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```

### STEP 01- Create a virtual environment after opening the repository

```bash
python -m venv medical_chatbot

```

```bash
medical_chatbot\Scripts\activate
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your GROQ credentials as follows:

```ini
GROQ_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

```


```bash
# run the following command to creates a searchable database from PDF files for building a question-answering system.
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```




