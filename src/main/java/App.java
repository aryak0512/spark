import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class App {

    private static final Logger log = LoggerFactory.getLogger(App.class);

    public static void main(String[] args) {

        // create the context
        SparkConf conf = getConfig();
        JavaSparkContext sparkContext = new JavaSparkContext(conf);

        List<Double> nums = new ArrayList<>();
        nums.add(3425.89);
        nums.add(425.89);
        nums.add(345.89);

        // build a RDD
        JavaRDD<Double> parallelize = sparkContext.parallelize(nums);
        long count = parallelize.count();

        log.info("Count : {}", count);
        sparkContext.close();
    }

    private static SparkConf getConfig() {
        // prepare the config
        SparkConf conf = new SparkConf();
        // sample label visible in reports
        conf.setAppName("my-spark app");
        // use all available cores of the system
        conf.setMaster("local[*]");
        return conf;
    }

}
