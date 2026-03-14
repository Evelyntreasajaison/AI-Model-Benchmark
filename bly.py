import streamlit as st
import time
import pandas as pd
from openai import OpenAI


st.set_page_config(
    page_title="AI Model Benchmark Dashboard",
    page_icon="",
    layout="wide"
)

st.title("AI Model Benchmark Dashboard")
st.markdown("Compare AI models based on **speed, usability, and response quality**.")


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=""  # Add your API key here. The key is not included in this repository because it is public.
"
)

models = {
    "Llama3-8B": {
        "id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "size": "8B Parameters"
    },
    "Llama3-70B": {
        "id": "meta-llama/Meta-Llama-3-70B-Instruct",
        "size": "70B Parameters"
    }
}


st.subheader("Enter Prompt")

prompt = st.text_area(
    "Prompt Input",
    placeholder="Type your prompt here... (Example: Explain Generative AI in simple terms)",
    height=120,
    label_visibility="collapsed"
)
run_button = st.button("Run Model Comparison")


if run_button:

    
    if prompt.strip() == "":
        st.warning("⚠️ Please enter a prompt before running the models.")
    else:

        results = []

        for model_name, info in models.items():

            with st.spinner(f"Running {model_name}..."):

                start = time.time()

                response = client.chat.completions.create(
                    model=info["id"],
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150
                )

                end = time.time()

                inference_time = round(end - start, 2)

                output = response.choices[0].message.content

                word_score = len(output.split())

                results.append({
                    "Model": model_name,
                    "Inference Speed (sec)": inference_time,
                    "Model Size": info["size"],
                    "Output Format": "Generated Text",
                    "Domain Performance Score": word_score,
                    "Output": output
                })

        df = pd.DataFrame(results)

        st.divider()

        col1, col2 = st.columns(2)

        col1.metric(
            "Fastest Model",
            df.sort_values("Inference Speed (sec)").iloc[0]["Model"]
        )

        col2.metric(
            "Highest Detail Score",
            df.sort_values("Domain Performance Score", ascending=False).iloc[0]["Model"]
        )

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Inference Speed")
            st.bar_chart(df.set_index("Model")["Inference Speed (sec)"])

        with col2:
            st.subheader("Domain Performance")
            st.bar_chart(df.set_index("Model")["Domain Performance Score"])

        st.divider()

        st.subheader("Model Comparison Table")
        st.dataframe(df.drop(columns=["Output"]))

        st.divider()

        st.subheader("💬 Model Responses")

        for r in results:
            with st.expander(f"{r['Model']} Response"):
                st.write(r["Output"])
