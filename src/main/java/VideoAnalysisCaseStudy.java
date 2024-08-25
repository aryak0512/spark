import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.time.Duration;
import java.time.temporal.TemporalUnit;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.apache.spark.sql.functions.*;

public class VideoAnalysisCaseStudy {
    public static void main(String[] args) throws InterruptedException {

        String path = "src/main/resources/casestudy/part-r-00000-d55d9fed-7427-4d23-aa42-495275510f78.csv";
        var sparkSession = Util.getSession();
        Dataset<Row> dataset = sparkSession.read()
                .option("header", true)
                .option("inferSchema", true) // since we need only numeric types
                .csv(path);

        dataset = dataset.withColumnsRenamed(Map.of("next_month_views","label"));


        // preprocessing -
        // 1. remove observation_date column
        // 2. ignore records which have
        // 3. replace null values in all_time_views, last_month_views, next_month_views by 0
        dataset = dataset.drop("observation_date");

        dataset = dataset.withColumn("all_time_views", when(col("all_time_views").isNull(), 0).otherwise(col("all_time_views")));
        dataset = dataset.withColumn("last_month_views", when(col("last_month_views").isNull(), 0).otherwise(col("last_month_views")));
        dataset = dataset.withColumn("label", when(col("label").isNull(), 0).otherwise(col("label")));

        dataset = dataset.filter("is_cancelled = false").drop(col("is_cancelled"));
        dataset.show();




        Thread.sleep(200000);
    }
}
