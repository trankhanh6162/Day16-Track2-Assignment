# Bao cao Lab 16 - GCP CPU LightGBM

1. Ha tang duoc trien khai bang Terraform tren GCP voi VM e2-medium tai us-central1-a.
2. VM chay Debian 12 trong private subnet va duoc truy cap bang SSH qua IAP.
3. Bo du lieu Credit Card Fraud Detection gom 284,807 giao dich va 492 giao dich gian lan.
4. Thoi gian tai du lieu la 3.44 giay va thoi gian huan luyen LightGBM la 10.07 giay.
5. Mo hinh dat AUC-ROC 0.8776, accuracy 0.9986 va F1-score 0.6385 tren tap test.
6. Precision dat 0.5913 va recall dat 0.6939; accuracy cao mot phan do du lieu mat can bang.
7. Do tre inference trung binh cho mot dong la 2.50 ms.
8. Throughput khi du doan 1,000 dong dat khoang 88,453 dong/giay tren CPU.
9. VM co 3.8 GiB RAM; sau benchmark con khoang 3.3 GiB kha dung.
10. Billing chua hien chi phi do du lieu cap nhat tre; tai nguyen duoc xoa sau khi thu thap bang chung.
