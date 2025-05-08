import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

# Đọc dữ liệu từ file CSV
df = pd.read_csv('normalized_data.csv')

# Bước 1: Tiền xử lý dữ liệu
# Tạo bảng xoay (pivot table) từ dữ liệu, nhóm theo 'MONTH' và 'ITEM TYPE'
# Sử dụng cột 'RETAIL SALES' để chuyển đổi thành dạng nhị phân (1/0)
basket = df.groupby(['MONTH', 'ITEM TYPE'])['RETAIL SALES'].sum().unstack().reset_index().fillna(0)

# Chuyển đổi thành giá trị True/False (1/0) cho các giá trị trong bảng
basket = basket.set_index('MONTH') > 0

# Bước 2: Áp dụng thuật toán apriori để khai thác tập mục thường xuyên
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)

# Bước 3: Tạo các luật kết hợp dựa trên các tập mục thường xuyên
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7,num_itemsets=1)

# Bước 4: Lọc các luật mạnh có confidence bằng ngưỡng (ví dụ: 1)
strong_rules = rules[rules['confidence'] == 1]

# Lưu các luật kết hợp vào tệp CSV
rules.to_csv('association_rules.csv', index=False, encoding='utf-8')
strong_rules.to_csv('strong_association_rules.csv', index=False, encoding='utf-8')

print("Các luật kết hợp đã được lưu vào tệp 'association_rules.csv'.")
print("Các luật mạnh đã được lưu vào tệp 'strong_association_rules.csv'.")
