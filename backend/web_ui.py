import os
import yaml
import mlflow
from dotenv import load_dotenv
import streamlit as st
import requests

load_dotenv()


if __name__ == "__main__":

    DEPLOY_HOST = os.getenv("deploy_host", "localhost")
    DEPLOY_PORT = os.getenv("deploy_port", 5001)
    API_HOST = os.getenv("host_server")
    API_PORT = os.getenv("api_port")
    st.title("🚀 Credit Risk ML Platform")

    uploaded = st.file_uploader("Load YAML config", type=["yaml", "yml"])

    if uploaded:
        cfg = yaml.safe_load(uploaded)

        st.subheader("📄 Loaded config yaml")
        st.json(cfg)

        if st.button("▶ Start learning"):
            with st.spinner("Model learns, wait..."):
                # run_id = run_main(cfg)

                uploaded.seek(0)

                files = {"file": (uploaded.name, uploaded, "application/x-yaml")}

                response = requests.post(
                    f"http://{API_HOST}:{API_PORT}/run_learning/",
                    files=files,
                    timeout=300  # 5 минут таймаут
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success("✅ Learning completed via FastAPI!")
                    st.write(f"**Response:**")
                    st.json(result)
                elif response.status_code == 422:
                    st.error("❌ Validation error in request")
                    st.write(f"Details: {response.text}")
                else:
                    st.error(f"❌ HTTP Error {response.status_code}")
                    st.write(f"Response: {response.text}")

            st.success("✅ Ready!")
            # st.write(f"Run ID: **{run_id}**")
            # st.write(f"[Open in MLflow](http://{DEPLOY_HOST}:{DEPLOY_PORT}/#/experiments/0/runs/{run_id})")