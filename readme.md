# AI Reader DEMO

## Dependencies and Installation
----------------------------
To install the MultiPDF Chat App, please follow these steps:

1. Clone the repository to your local machine. **And Install Python 3.9**.

2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```
3. Maybe you need to install:
   ```
   pip install tiktoken
   ```

4. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.
```commandline
OPENAI_API_KEY=your_secrit_api_key
```

## Usage
-----
To use the MultiPDF Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `app.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple PDF documents into the app by following the provided instructions.

5. Ask questions in natural language about the loaded PDFs using the chat interface.

## License
-------
The MultiPDF Chat App is released under the [MIT License](https://opensource.org/licenses/MIT).


## Buildx
1. Create a new Buildx builder which allows building for multiple platforms:
docker buildx create --name mybuilder --use

2. Inspect if your builder supports the required platforms: 
docker buildx inspect --bootstrap

3. Build and push the image using Buildx. Specify multiple platforms using the --platform flag:
docker buildx build --platform linux/amd64,linux/arm64,linux/arm/v7 -t username/face_analysis:tag --push .

