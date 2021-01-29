from flask import Flask, render_template, request
import os
from Vessel import *
from typing import Optional
import jinja2
from rem import *
from ex_hem import *
from grade import *

app = Flask(__name__)


@app.route('/upload')  
def upload():  
    return render_template("upload.html")  

@app.route('/index')  
def ind():  
    return render_template("index.html")  

@app.route('/about')  
def abt():  
    return render_template("about.html")  

@app.route('/nethra')  
def neth():  
    return render_template("nethra_ai.html")  

@app.route('/research')  
def res():  
    return render_template("research.html")  
    
@app.route('/vessel/<img>')
def bv(img):
    x=vessel_seg(img)
    x=[str(img)]
    return render_template("vessel.html", fname=x)
    
@app.route("/segment/<img>")
def seg(img):
    od_gen(img)
    hem_gen(img)
    exd_gen(img)
    # vessel(img)
    x=[str(img)]
    ex_clean(img)
    hem_clean(img)
    grd=[str(predg(img)[0])]

    return render_template('dr.html', fname=x, gr=grd)

@app.route("/glaucoma/<img>")
def gla(img):
    x=od_gen(img)
    y=cup_gen(img)
    yd,xd=identify("static/"+img[:-4]+"_od.jpg")
    yc,xc=identify("static/"+img[:-4]+"_cup.jpg")
    ratio=yc/yd
    if ratio>=0.6:
        c="Yes"
        if ratio>0.7:
            sev="Severe"
        else:
            sev="Mild"
    else:
        c="No"
        sev="Normal"
    im=[str(img)]
    ir=[str(sev)]
    return render_template('glaucoma.html', fname=im, result=ir)


if __name__ == '__main__':
    app.run(debug = True)
