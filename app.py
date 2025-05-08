import pandas as pd
from flask import Flask, render_template, request

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# Đọc dữ liệu từ file CSV
df = pd.read_csv('normalized_data.csv')
association_rules_df = pd.read_csv('strong_association_rules.csv')  # Đọc các luật kết hợp từ file CSV

# Hàm để áp dụng các luật kết hợp đã lưu cho loại sản phẩm dự đoán
def get_association_rules_for_item_type(item_type):
    # Lọc các luật liên quan đến loại sản phẩm dự đoán
    related_rules = association_rules_df[association_rules_df['antecedents'].apply(lambda x: item_type in x)]

    # Lấy danh sách các loại sản phẩm gợi ý mua kèm từ cột 'consequents'
    recommended_item_types = set()
    for _, rule in related_rules.iterrows():
        recommended_item_types.update(eval(rule['consequents']))  # Chuyển đổi string 'consequents' thành set

    # Trả về danh sách các loại sản phẩm gợi ý mua kèm
    return list(recommended_item_types)

# Hàm thống kê doanh thu bán lẻ và tồn kho của các loại sản phẩm
def get_sales_statistics_for_suggested_items(month, suggested_item_types):
    # Lọc dữ liệu theo tháng
    month_data = df[df['MONTH'] == month]
    
    # Thống kê doanh thu bán lẻ và doanh thu tồn kho cho các loại sản phẩm gợi ý
    item_sales = month_data[month_data['ITEM TYPE'].isin(suggested_item_types)].groupby('ITEM TYPE').agg({
        'RETAIL SALES': 'sum',
        'WAREHOUSE SALES': 'sum'
    }).reset_index()

    return item_sales

# Hàm đánh giá sản phẩm nào nên khuyến mãi
def get_promotion_item(item_sales):
    # Sản phẩm nào có doanh thu bán lẻ và kho thấp nhất sẽ được khuyến mãi
    low_sales_items = item_sales[(item_sales['RETAIL SALES'] == item_sales['RETAIL SALES'].min()) & 
                                 (item_sales['WAREHOUSE SALES'] == item_sales['WAREHOUSE SALES'].min())]

    # Nếu có sản phẩm có doanh thu thấp nhất thì lấy sản phẩm đó
    promotion_item = low_sales_items['ITEM TYPE'].values[0] if not low_sales_items.empty else 'Không có mặt hàng phù hợp'

    return promotion_item

# Route chính cho ứng dụng web
@app.route('/')
def index():
    return render_template('index1.html')

# Route để xử lý dữ liệu và dự đoán kết quả khi người dùng nhập vào tháng và tên loại sản phẩm
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu đầu vào từ form
        month = int(request.form['month'])
        item_type = request.form['item_type']  # Nhập tên loại sản phẩm

        # Kiểm tra điều kiện đầu vào (tháng hợp lệ)
        if not (1 <= month <= 12):
            return "Tháng không hợp lệ! Vui lòng nhập từ 1 đến 12."

        # Lấy các loại sản phẩm gợi ý mua kèm từ các luật kết hợp đã lưu
        suggested_item_types = get_association_rules_for_item_type(item_type)

        # Thống kê doanh thu bán lẻ và doanh thu tồn kho cho các loại sản phẩm gợi ý
        item_sales = get_sales_statistics_for_suggested_items(month, suggested_item_types)

        # Đánh giá sản phẩm nào nên được khuyến mãi
        promotion_item = get_promotion_item(item_sales)

        # Trả về kết quả
        return render_template('result.html', 
                               item_type=item_type,
                               suggested_item_types=suggested_item_types, 
                               item_sales=item_sales.to_dict(orient='records'),
                               promotion_item=promotion_item)

    except ValueError:
        return "Dữ liệu nhập vào không hợp lệ. Vui lòng kiểm tra lại."

if __name__ == '__main__':
    app.run(debug=True)
