import tabula
import pandas as pd

# to plot csv files easily:
import plotly
import plotly.plotly as py
# import plotly.graph_objs as go
import plotly.figure_factory as FF
plotly.tools.set_credentials_file(username='mawanda', api_key='1FMxcIoMD5xFR2Nf3Zwf')

# df = tabula.read_pdf(
# 	input_path='./PDFs/polizza.pdf',
# 	pages='5-5',
# 	encoding='latin',
# 	java_options='-Dsun.java2d.cmm=sun.java2d.cmm.kcms.KcmsServiceProvider'
# )
# print(df)
# tabula.convert_into(
# 	input_path='./polizza.pdf',
# 	output_path='output.csv',
# 	output_format='csv',
# 	java_options='-Dsun.java2d.cmm=sun.java2d.cmm.kcms.KcmsServiceProvider',
# 	pages='5-5')

tabula.convert_into('./PDFs/polizza.pdf', output_path='./output.csv', output_format='csv', pages='all')

# df = pd.read_csv('output.csv', encoding='latin')
# data_table = FF.create_table(df.head())
# py.iplot(data_table, filename='data-table')


