import org.apache.spark.SparkConf;
import org.apache.spark.sql.SparkSession;

public class Util {

    /**
     * Creates the spark conf object
     * @return the spark conf object
     */
    public static SparkConf getConfig() {
        // prepare the config
        SparkConf conf = new SparkConf();
        // sample label visible in reports
        conf.setAppName("my-spark app");
        // use all available cores of the system
        conf.setMaster("local[*]");
        return conf;
    }

    /**
     * creates the spark session object used by Spark SQL API
     * @return the spark session
     */
    public static SparkSession getSession() {
        return SparkSession.builder()
                .appName("my-app")
                .master("local[*]")
                .getOrCreate();
    }
}
