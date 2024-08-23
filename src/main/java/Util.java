import org.apache.spark.SparkConf;

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

}
