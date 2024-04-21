import glob
import librosa

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd

from dash import html, dcc, Input, Output
from assets.dash_css.dashboard_css import CONTENT_STYLE
from server import app

NavBar_Layout = dbc.NavbarSimple(
    children=[
        dbc.NavItem(
            dbc.NavLink(
                children=[html.Span('About Project')], href='/', style={'margin-top': '3px', 'margin-right': '10px'})),

        dbc.NavItem(dbc.NavLink(
                    children=[html.Span('Exploratory Data Analysis')], href='/eda-page', style={'margin-top': '3px', 'margin-right': '10px'})),

        dbc.NavItem(dbc.NavLink(
                    children=[html.Span('Model Training')], href='/model-training-page', style={'margin-top': '3px', 'margin-right': '10px'})),

        dbc.NavItem(dbc.NavLink(
                    children=[html.Span('Results')], href='#', style={'margin-top': '3px', 'margin-right': '10px'})),

        dbc.NavItem(dbc.NavLink(
                    children=[html.Span('Demonstration')], href='#', style={'margin-top': '3px', 'margin-right': '20px'}))
    ],

    brand=[
        html.Span(
            'SpeakSense', style={'font-size': '1.2rem', 'margin-right': '10px'}),
        html.Span('Machine Learning CSCI 6364',
                  style={'font-size': '0.7rem'})
    ],

    color='dark', dark=True, brand_href='/', brand_style={'font-weight': 'bold', 'margin-left': '20px'}, fluid=True, sticky=True,

)

UI_Content_Layout = html.Div(id='page-content', style=CONTENT_STYLE)

Dashboard_Layout = html.Div([
    html.Div(
        dbc.Container(
            [
                html.H1('SpeakSense - AI Language Detection Tool',
                        className='display-3'),

                html.Span(
                    'Created By: Abde Manaaf (G29583342), Gehna Ahuja (G35741419), Venkatesh Shanmugam (G00000000)',
                    className='lead'),

                html.Hr(className='my-2'),

                html.Span(html.H4('Project Objective',
                          style={'margin-top': '25px'})),
                html.Br(),

                html.P([
                    html.Ul([
                        html.Li(html.Span(
                            html.H5('The objective of this project is to develop a robust and accurate system capable of detecting the language spoken in audio recordings.'))),

                        html.Li(html.Span(html.H5('By leveraging advanced machine learning algorithms and signal processing techniques, the system aims to accurately identify the language '
                                                  'spoken in various audio inputs, spanning diverse accents, dialects, and environmental conditions. '))),

                        html.Li(html.Span(html.H5('This language detection solution seeks to provide practical applications in speech recognition, transcription, translation, and other fields requiring language-specific processing, '
                                                  'thereby enhancing accessibility and usability across linguistic boundaries.')))
                    ])
                ], style={'margin-bottom': '50px'}),

                html.Hr(className='my-2'),

                html.Span(html.H4('Data Sources',
                          style={'margin-top': '25px'})),
                html.Br(),

                html.P([
                    html.Span(
                        html.H5('In sourcing data, we aim to gather diverse and representative datasets that encompass a wide range of languages, accents, dialects, and speaking styles. We shall be using data sets taken from Kaggle.')),

                    html.Br(),

                    dbc.ListGroup(
                        [
                            dbc.ListGroupItem(
                                html.A('Audio Dataset with 10 Indian Languages (kaggle.com) - Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu, Urdu',
                                       href='https://www.kaggle.com/datasets/hbchaitanyabharadwaj/audio-dataset-with-10-indian-languages', target='_blank')),
                            dbc.ListGroupItem(
                                html.A('Spoken Language Identification (kaggle.com) - English, Spanish and German.',
                                       href='https://www.kaggle.com/datasets/toponowicz/spoken-language-identification', target='_blank')),
                            dbc.ListGroupItem(
                                'As a Test (Holdout) Set we have collected Audio from our Friends, Family and Faculty to Demonstrate Live Prediction.'),
                        ],
                        flush=True,
                    ),
                ], style={'margin-bottom': '50px'}),

                html.P([
                    html.A(dbc.Button('GitHub Repository!', color='primary'),
                           href='https://github.com/default741/speak-sense',  target='_blank', style={'margin-right': '20px'}),

                    html.A(dbc.Button('Next: Exploratory Data Analysis', color='primary'),
                           href='/eda-page'),
                ], className='lead'),
            ],
            fluid=True, className='py-3'
        ), className='p-3 bg-light rounded-3'
    )
])

side_pannel_layout = html.Div([
    dbc.Nav(
        [
            dbc.NavItem(dbc.NavLink("Amplitude Plots",
                        href="#amplitude-plots", id='link-amplitude-plot', n_clicks=0)),
            dbc.NavItem(dbc.NavLink("Spectogram Plots",
                        href="#spectogram-plots", id='link-spectogram-plot', n_clicks=0)),
            dbc.NavItem(dbc.NavLink("Descriptive Statistics",
                        href="#descriptive-stats", id='link-descriptive-stats')),
        ],
        vertical="md",
    )
])


def parse_audio_amplitude_plot(file_name: str) -> object:
    """Plots Amplitude vs Time for an Audio File.

    Args:
        file_name (str): File Path to an audio file.

    Returns:
        object: Amplitude Plot
    """

    audio_data, sample_rate = librosa.load(
        file_name, sr=None)

    language = file_name.split('/')[-1].split('.')[0].capitalize()

    duration = librosa.get_duration(y=audio_data, sr=sample_rate)
    time_samples = np.linspace(0, duration, len(audio_data))

    waveform_plot = go.Figure()
    waveform_plot.add_trace(go.Scatter(
        x=time_samples, y=audio_data, mode='lines'))
    waveform_plot.update_layout(
        title=f'Waveform (Amplitude) Plot ({language})',
        xaxis={'title': 'Time [seconds]'},
        yaxis={'title': 'Amplitude'}
    )

    return waveform_plot


def parse_audio_spectogram_plot(file_name: str) -> object:
    """Plots Amplitude vs Time for an Audio File.

    Args:
        file_name (str): File Path to an audio file.

    Returns:
        object: Amplitude Plot
    """

    audio_data, _ = librosa.load(file_name, sr=None)

    language = file_name.split('/')[-1].split('.')[0].capitalize()

    stft_transform = librosa.stft(audio_data)
    amplitude_to_db = librosa.amplitude_to_db(
        np.abs(stft_transform), ref=np.max)

    spectrogram_plot = go.Figure()
    spectrogram_plot.add_heatmap(x=np.arange(
        amplitude_to_db.shape[1]), y=np.arange(amplitude_to_db.shape[0]), z=amplitude_to_db)
    spectrogram_plot.update_layout(
        title=f'Spectrogram Plot ({language})',
        xaxis={'title': 'Time [seconds]'},
        yaxis={'title': 'Frequency [hertz]'},
        showlegend=False
    )

    return spectrogram_plot


EDA_Layout = html.Div([
    dbc.Container(
        [
            html.H1('Exploratory Data Analysis (EDA)',
                    className='display-3'),

            html.Span(
                'Exploratory Data Analysis (EDA) is a statistical approach used to analyze and visualize data sets to understand their main characteristics, uncover patterns, and identify potential relationships between variables.',
                className='lead'),

            html.Hr(className='my-2'),

            html.P([
                html.Ul([
                    html.Li(html.Span(
                        html.H5('Now while looking at the Indian Languages Audio Files, we notices that the audio files were of different durations. So to not create any '
                                'bias on time duration of audio files, we combined each language audio files into one file and chunked that file to 10 seconds duration audio files.'))),

                    html.Li(html.Span(html.H5('The Average Audio file size is 108 kB centering around 78 kB. All audio files have the same Sample rate of 22050 hertz. (Sample rate refers '
                                              'to the number of samples of audio carried per second, measured in Hertz (Hz). Higher sample rates generally result in higher audio quality but also larger file sizes.)'))),

                    html.Li(html.Span(html.H5(
                        'A Spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies with time.'))),

                    html.Li(html.Span(html.H5(
                        'An Amplitude Plot, also known as a waveform plot, displays the amplitude of an audio signal as a function of time.')))
                ])
            ], style={'margin-bottom': '50px', 'margin-top': '25px'}),

            dbc.Row([
                dbc.Col(side_pannel_layout, width=2),
                dbc.Col([
                    dbc.Row([
                        dbc.Col(dcc.Graph(figure=parse_audio_amplitude_plot(
                            file_name='./layouts/eda_audio_data/english.flac'))),
                        dbc.Col(dcc.Graph(figure=parse_audio_amplitude_plot(
                            file_name='./layouts/eda_audio_data/german.flac'))),
                        dbc.Col(dcc.Graph(figure=parse_audio_amplitude_plot(
                            file_name='./layouts/eda_audio_data/spanish.flac')))
                    ]),
                ], width=10, id='eda-content-container')
            ], style={'margin-top': '25px'}),

            html.P([
                html.A(dbc.Button('Next: About Page', color='primary'),
                       href='/', style={'margin-right': '20px'}),

                html.A(dbc.Button('Next: Model Training', color='primary'),
                       href='/model-training-page'),
            ], className='lead', style={'margin-top': '25px'}),
        ],
        fluid=True, className='py-3',
    )
], className='p-3 bg-light rounded-3')


Model_Training_Layout = html.Div([
    dbc.Container([
        html.H1('Model Training',
                className='display-3'),

    ], fluid=True, className='py-3')
], className='p-3 bg-light rounded-3')


@app.callback(
    [
        Output("eda-content-container", "children"),
        Output("link-amplitude-plot", "n_clicks"),
        Output("link-spectogram-plot", "n_clicks"),
        Output("link-descriptive-stats", "n_clicks")
    ],
    [
        Input("link-amplitude-plot", "n_clicks"),
        Input("link-spectogram-plot", "n_clicks"),
        Input("link-descriptive-stats", "n_clicks")
    ]
)
def change_eda_content(amplitude_plot_click: int, spectogram_plot_click: int, descriptive_stats_click: int):
    if amplitude_plot_click:
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=parse_audio_amplitude_plot(
                file_name='./layouts/eda_audio_data/english.flac'))),
            dbc.Col(dcc.Graph(figure=parse_audio_amplitude_plot(
                file_name='./layouts/eda_audio_data/german.flac'))),
            dbc.Col(dcc.Graph(figure=parse_audio_amplitude_plot(
                file_name='./layouts/eda_audio_data/spanish.flac')))
        ]), 0, 0, 0

    if spectogram_plot_click:
        return dbc.Row([
            dbc.Col(dcc.Graph(figure=parse_audio_spectogram_plot(
                file_name='./layouts/eda_audio_data/english.flac'))),
            dbc.Col(dcc.Graph(figure=parse_audio_spectogram_plot(
                file_name='./layouts/eda_audio_data/german.flac'))),
            dbc.Col(dcc.Graph(figure=parse_audio_spectogram_plot(
                file_name='./layouts/eda_audio_data/spanish.flac')))
        ]), 0, 0, 0

    if descriptive_stats_click:
        language_dataframe_v1 = pd.read_csv(
            './layouts/eda_audio_data/language_dataframe_v1.csv')
        language_dataframe_v2 = pd.read_csv(
            './layouts/eda_audio_data/language_dataframe_v2.csv')

        lang_value_counts = pd.merge(
            left=pd.DataFrame(
                language_dataframe_v1['language_label'].value_counts()).reset_index(),
            right=pd.DataFrame(
                language_dataframe_v2['language_label'].value_counts()).reset_index(),
            on='language_label').rename(columns={'language_label': 'Language Labels', 'count_x': 'Before Chunking', 'count_y': 'After Chunking'})

        audio_duration = language_dataframe_v1['audio_duration_sec'].value_counts(
        )
        audio_duration = pd.DataFrame(
            {'Audio Duration (Seconds)': audio_duration.index, 'Counts': audio_duration.values})

        return dbc.Row([
            dbc.Col(width=1),
            dbc.Col(dbc.Table.from_dataframe(audio_duration.sort_values(by=['Audio Duration (Seconds)']),
                    striped=True, bordered=True, hover=True)),
            dbc.Col(width=1),
            dbc.Col(dbc.Table.from_dataframe(lang_value_counts,
                    striped=True, bordered=True, hover=True)),
            dbc.Col(width=1),
        ]), 0, 0, 0

    return dbc.Row([
        dbc.Col(dcc.Graph(figure=parse_audio_amplitude_plot(
                file_name='./layouts/eda_audio_data/english.flac'))),
        dbc.Col(dcc.Graph(figure=parse_audio_amplitude_plot(
                file_name='./layouts/eda_audio_data/german.flac'))),
        dbc.Col(dcc.Graph(figure=parse_audio_amplitude_plot(
                file_name='./layouts/eda_audio_data/spanish.flac')))
    ]), 0, 0, 0
