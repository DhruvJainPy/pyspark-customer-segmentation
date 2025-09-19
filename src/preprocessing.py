from utils import create_spark_session
from pyspark.sql.functions import col, count, when, isnan, round, to_timestamp, lower, regexp_replace, trim, date_format

# Create Spark session.
sparkSession = create_spark_session("CustomerSegementation")

# Load data from CSV file.
def load_data(file_path):
    data = sparkSession.read.csv(file_path, header = True, inferSchema = True)
    return data

data = load_data("data/OnlineRetail.csv")

# Print no of rows and columns.
print(data.count())
print(data.columns)

# Drop rows with null values, duplicates and filter out rows with negative or zero quantity and unit price.
data.select([count(when(col(c).isNull() | isnan(c), c)).alias(c) for c in data.columns]).show()
data = data.dropna()
data = data.dropDuplicates()
data = data.filter((col("Quantity") > 0) & (col("UnitPrice") > 0))

# Print no of rows after cleaning the null values and duplicates.
print(data.count())

# Feature engineering and data type conversion.
data = data.withColumn("InvoiceDate", to_timestamp(col("InvoiceDate"), "M/d/yy H:mm")) \
           .withColumn("CustomerID", col("CustomerID").cast("string")) \
           .withColumn("Quantity", col("Quantity").cast("int")) \
           .withColumn("UnitPrice", col("UnitPrice").cast("float")) \
           
data = data.withColumn('TotalAmount', round(col('Quantity') * col('UnitPrice'),2))
data = data.withColumn("MonthYear", date_format(col("InvoiceDate"), "yyyy-MM"))

data = data.withColumn("Description", lower(trim(col("Description"))))
data = data.withColumn("Description", regexp_replace(col("Description"), "[^a-zA-Z0-9 ]", ""))

# Show final cleaned data.
data.show(10)

data.write.mode("overwrite").parquet("data/online_retail.parquet")
sparkSession.stop()
