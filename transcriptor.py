import datetime
import numpy as np
import os
import pandas as pd
import subprocess
import streamlit as st
import torch
import whisper

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# Embedding model
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Whisper model
whisper_model = whisper.load_model("base")

# Global Streamlit settings
st.set_page_config(layout="wide", page_title="Transkription")

with open("./css/style_purple.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
        <style>
            [data-testid="stHeader"]::before {
                content: "ai4ki";
                font-family: Arial, sans-serif;
                font-weight: bold;
                font-size: 40px;
                color: #3b1f82;
                position: relative;
                left: 30px;
                top: 10px;
            }
        </style>
        """,
    unsafe_allow_html=True,
)

# Define page layout
cols = st.columns([2, 0.2, 4])


def save_uploaded_file(uploaded_file):
    try:
        with open(uploaded_file.name, 'wb') as f_audio:
            f_audio.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


with cols[0]:
    st.markdown("### Lade eine Audiodatei hoch:")
    audiofile = st.file_uploader(label="Lade eine Audiodatei hoch:",
                                 type=['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'webm'],
                                 label_visibility="collapsed")
    n_speakers = st.slider("**Anzahl der Sprecher_innen:**", 1, 5, 2)
    st.markdown("")
    start_transcribe = st.button("Audiodatei transkribieren")


def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))


def transcribe_diarize(wave_file, duration, num_speakers):

    response = whisper_model.transcribe(wave_file, language="de")
    segments = response["segments"]

    try:
        def segment_embedding(segment):
            audio = Audio()
            start = segment["start"]
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio.crop(wave_file, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)
        embeddings = np.nan_to_num(embeddings)

        if num_speakers == 0:
            score_num_speakers = {}

            for num_speakers in range(2, 10 + 1):
                clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
                score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
                score_num_speakers[num_speakers] = score
            best_num_speaker = max(score_num_speakers, key=lambda x: score_num_speakers[x])
        else:
            best_num_speaker = num_speakers

        clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'Sprecher_in ' + str(labels[i] + 1)

        objects = {
            'Start': [],
            'Ende': [],
            'Sprecher_in': [],
            'Text': []
        }
        text = ''
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(convert_time(segment["start"])))
                objects['Sprecher_in'].append(segment["speaker"])
                if i != 0:
                    objects['Ende'].append(str(convert_time(segments[i - 1]["end"])))
                    objects['Text'].append(text)
                    text = ''
            text += segment["text"] + ' '
        objects['Ende'].append(str(convert_time(segments[i - 1]["end"])))
        objects['Text'].append(text)

        df_results = pd.DataFrame(objects)

        diarized_text = ""
        for _, rows in df_results.iterrows():
            diarized_text += f"{rows.Sprecher_in} [{rows.Start}-{rows.Ende}]:\n{rows.Text.strip()}\n\n"

        return diarized_text
    except Exception as e:
        raise RuntimeError("Error Running inference with local model", e)


with cols[2]:
    if start_transcribe:
        if audiofile is not None:
            if save_uploaded_file(audiofile):
                wave_filename = "input_audio.wav"
                os.system(f'ffmpeg -i "{audiofile.name}" -ar 16000 -ac 1 -c:a pcm_s16le "{wave_filename}"')
                output = subprocess.check_output(['ffprobe',
                                                  '-i', 'input_audio.wav', '-show_entries',
                                                  'format=duration', '-v', 'quiet',
                                                  '-of', 'csv=%s' % "p=0"])

                st.markdown("### Transkript:")
                with st.spinner(f"Transkription von {audiofile.name} läuft..."):
                    transcription = transcribe_diarize(wave_filename, float(output), n_speakers)

                st.text_area(label="Transkription:",
                             value=transcription, height=500,
                             label_visibility="collapsed")
                st.markdown("")
                st.download_button(label="Transkription herunterladen",
                                   data=transcription,
                                   file_name=f"Transcript_{audiofile.name}.txt")

                os.remove("input_audio.wav")
                os.remove(audiofile.name)

        else:
            st.warning(f"Beim Upload ist etwas schief gegangen -- versuche es noch einmal!")
