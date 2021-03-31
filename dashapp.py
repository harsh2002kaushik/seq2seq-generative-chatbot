#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 10:45:19 2021

@author: harsh
"""


import time

import dash
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import train

model = train.Model()
#model.restore()
model.load_weights('model_weights')
#model.save_embeddings()
#embedding.get_weights()
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def textbox(text, box = 'other'):
    style = {
        'max_width':'55%',
        'width':'max-content',
        'padding': '10px 15px',
        'border-radius':'25px'}
    
    if box == 'self':
        style['margin-left'] = 'auto'
        style['margin-right'] = 0
        color = 'primary'
        inverse = True
        
    elif box == 'other':
        style['margin-left'] = 0
        style['margin-right'] = 'auto'
        color = 'light'
        inverse = False
        
    else:
        raise ValueError('Incorrect')
        
    return dbc.Card(text, style=style, body=True, color=color, inverse=inverse)



app.layout = dbc.Container(
    fluid = True,
    children=[
        html.H1('chatbot'),
        html.Hr(),
        dcc.Store(id='store-conversation',data=''),
        html.Div(
            style = {
                'width': '80%',
                'max_width':'800px',
                'height':'70vh',
                'margin':'auto',
                'overflow-y':'auto'},
            id='display-conversation'
            ),
        dbc.InputGroup(
            style = {'width':'80%','max_width':'800px','margin':'auto'},
            children = [
                dbc.Input(id='user-input', placeholder = 'write to the chatbot...',type='text'),
                dbc.InputGroupAddon(dbc.Button('Submit',id='submit'),addon_type='append')])])

@app.callback(
    Output('display-conversation','children'),[Input('store-conversation','data')])

def update_display(chat_history):
    return[
        textbox(x,box='other') if i%2 == 0 else textbox(x,box='self')
        for i,x in enumerate(chat_history.split('<end of line>'))]

@app.callback(
    [Output('store-conversation','data'), Output('user-input','value')],
    [Input('submit','n_clicks'), Input('user-input','n_submit')],
    [State('user-input','value'), State('store-conversation','data')])

def run_chatbot(n_clicks, n_submit, user_input, chat_history):
    if n_clicks == 0:
        return '',''
    
    if user_input is None or user_input == '':
        return chat_history,''
    
    else:
        chat_history = chat_history +'<end of line>'+ user_input +'<end of line>'+ model.evaluate(user_input)[0]
        return chat_history,''


if __name__=='__main__':
    PORT = 3000
    app.run_server(port = PORT)
