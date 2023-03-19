import gradio as gr
import pandas as pd
import numpy as np

with gr.Blocks() as demo:
    with gr.Row():
        # inputs
        with gr.Column(scale=2):
            file_name = gr.Dropdown(label="Podaj nazwe pliku",
                                    choices=["australian.txt", "diabetes.txt", "car.txt", "fertilityDiagnosis.txt",
                                             "german-credit.txt", "heartdisease.txt", "hepatitis-filled.txt",
                                             "house-votes-84.txt", "mushroom-modified-filled.txt", "nursery.txt",
                                             "soybean-large-all-filled.txt", "SPECT-all-modif.txt",
                                             "SPECTF-all-modif.txt"])
            rows_number = gr.Number(label="Podaj numer wierszy", precision=0)
            question_input = gr.Dropdown(label="Pytanie",
                                         choices=["Ile klas decyzyjnych?", "Wielkość każdej klasy decyzyjnej"])
            submit = gr.Button("Submit")

        # outputs
        with gr.Column(scale=5):
            output_textbox = gr.Textbox(interactive=False, label="Informacje o pliku")
            output_textbox2 = gr.Textbox(interactive=False, label="Odpowiedź na pytanie")
            board = gr.Dataframe(interactive=False, overflow_row_behaviour="show_ends")


    def open_txt(file, rows, q_input):
        # logika do wybranego pliku
        file_path = "../dane/" + file
        df = pd.read_csv(file_path, sep=' ', header=None)
        df.replace(" ", np.nan, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)
        objects_size = len(df.index)
        attribute_size = len(df.columns)
        result_text = f"Objects: {objects_size}\nAttributes: {attribute_size}"
        if rows != 0:
            result_df = df.iloc[0:rows]
        else:
            result_df = df

        # logika do klas decyzyjnych
        dfc = pd.read_csv("../dane/_info-data-discrete.txt", sep=' ', header=None)
        dfc_size = len(dfc.index)
        if q_input == question_input.choices[0]:
            question_result = f"{dfc_size}"
        elif q_input == question_input.choices[1]:
            temp_result = ""
            for row_index in range(dfc_size):
                temp_result += f"{dfc.iat[row_index, 0]} = {dfc.iat[row_index, 2]}\n"
            question_result = temp_result
        else:
            question_result = "None"

        return [result_df, result_text, question_result]


    submit.click(open_txt, inputs=[file_name, rows_number, question_input], outputs=[board, output_textbox,
                                                                                     output_textbox2])

demo.launch(server_port=7861)