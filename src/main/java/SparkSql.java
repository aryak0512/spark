import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SparkSql {

    private static final Logger log = LoggerFactory.getLogger(SparkSql.class);

    public static void main(String[] args) {

        SparkSession sparkSession = SparkSession.builder()
                .appName("my-app")
                .master("local[*]")
                .getOrCreate();

        String path = "src/main/resources/students.csv";
        Dataset<Row> dataset = sparkSession.read()
                .option("header", true)
                .csv(path);

        // work internally as a distributed RDD
        dataset.show();
        var count = dataset.count();
        log.info("Count of rows : {}", count);
        sparkSession.close();
    }
}
