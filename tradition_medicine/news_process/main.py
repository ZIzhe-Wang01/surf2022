from File_process import *
import pandas as pd
import csv
import xlrd
import codecs


def xlsx_to_csv(xlsx_file):
    workbook = xlrd.open_workbook(xlsx_file)
    table = workbook.sheet_by_index(0)
    with codecs.open(r'G:\surf\tradition_medicine\Data\new_csv.csv', 'w', encoding='utf-8') as f:
        write = csv.writer(f)
        for row_num in range(table.nrows):
            row_value = table.row_values(row_num)
            write.writerow(row_value)





if __name__ == '__main__':

  print("Begin")
  #path = r"" \
    #     r"G:\\SURF\\tradition_medicine\\Data\\temp\\"
  dest_path = r"G:\\SURF\\tradition_medicine\\Data\\temp_news\\"
  # transform_document(path, dest_path)

  out_path = r'G:\surf\tradition_medicine\Data'  #路径为会议通知文件夹和 Excel 模板所在的位置，可按实际情况更改
  in_path = r'\Coding spreadsheet - Tone+frame.xlsx'


  files_list = read_total_files(dest_path)
  # for each in files_list:
  #   print(each)


  code_list, news_dir_title = get_title(files_list)
  # for each in news_dir_title.keys():
  #     print(each + " title: " + news_dir_title[each])
  # for each in code_list:
  #     print(each)


  news_dir_content = get_content(files_list)
  # for each in news_dir_content.keys():
  #  print(each + " content: " )
  #  print(news_dir_content[each])
  insert_article_code(out_path, in_path, code_list, news_dir_title, news_dir_content)

  #xlsx_file = r'G:\surf\tradition_medicine\Data\new_sheet.xlsx'
  #xlsx_to_csv(xlsx_file)