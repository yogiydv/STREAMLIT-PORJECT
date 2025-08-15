
# Alzheimer's Detection (GitHub-ready, model downloads at runtime)

This repository contains a Streamlit demo app that will attempt to download a pretrained model at startup
if it is not present in `model/alzheimer_model.h5`. This allows you to keep the GitHub repo small and
store the large model file elsewhere (Google Drive, Hugging Face, etc.).

## How to use

1. Create a new GitHub repo and push this project.
2. In Streamlit Community Cloud, when you deploy the app, set the secret `MODEL_URL` to the direct download URL of your `alz_model.h5` file:
   - Go to your app -> Settings -> Secrets -> Add `MODEL_URL` : `https://.../alz_model.h5`
   - Or you can set DEFAULT_MODEL_URL in app.py before uploading.
3. Deploy on Streamlit Cloud pointing to `app.py`.

## Local usage
- Optionally, place `model/alzheimer_model.h5` in the `model/` folder.
- Install dependencies:
```
pip install -r requirements.txt
```
- Run:
```
streamlit run app.py
```

## Notes
- If you don't have a pretrained model, the app will run in a **demo heuristic mode** that is not medically valid.
- For large model files, host them on Google Drive, Hugging Face, or S3 and provide a direct download link in `MODEL_URL`.
