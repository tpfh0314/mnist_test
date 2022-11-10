from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

import predict_image

app = Flask(__name__)

format_list = ['jpg','jpeg','png']

@app.route('/file_upload', methods=['GET', 'POST'])
def file_upload():
    if request.method == 'POST':
        f = request.files['file']
        file = False
        for format in format_list:
            if format in f.filename.split('.')[1]:
                file = True
            if(file):
                f.save(secure_filename(f.filename))

        img_list, ans_df = predict_image.predict_image(f.filename)

        print(ans_df)
        return ans_df.to_html()
    else:
        return render_template('upload.html')


if __name__ == '__main__':
    app.run(port="3014", debug=True)