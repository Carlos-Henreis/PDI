from classifier import ImageClassifier

model_name = '0008_CH3F'

# treinamento do modelo
ic = ImageClassifier()
ic.set_classes(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
ic.set_model_name(model_name)

# use the model to classify images from a different folder
ic.predict('data/{}/predict/'.format(model_name))
ic.print_results()

# create an html page with a graphic report
ic.print_html_report()
