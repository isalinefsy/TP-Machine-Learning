import csv

with open("file.csv") as my_csv_file:
  header_and_data = list(csv.reader(my_csv_file, delimiter=','))

header = header_and_data[0]
data = header_and_data[1:]
