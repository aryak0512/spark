import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class App {

    private static final Logger log = LoggerFactory.getLogger(App.class);

    public static void main(String[] args) {

        // create the context
        SparkConf conf = Util.getConfig();
        JavaSparkContext sparkContext = new JavaSparkContext(conf);

        List<Double> nums = new ArrayList<>();
        nums.add(3425.89);
        nums.add(425.89);
        nums.add(345.89);

        exploreCount(nums, sparkContext);
        exploreReduce(nums, sparkContext);

        sparkContext.close();
    }

    /**
     * Simple count operation
     * @param nums the dataset to operate on
     * @param sparkContext the context
     */
    private static void exploreCount(List<Double> nums, JavaSparkContext sparkContext) {
        // build a RDD
        JavaRDD<Double> rdd = sparkContext.parallelize(nums);
        log.info("Count : {}", rdd.count());
    }

    /**
     * Simple reduce operation, demonstrating addition of elements
     * @param nums the dataset to operate on
     * @param sparkContext the context
     */
    private static void exploreReduce(List<Double> nums, JavaSparkContext sparkContext) {
        // build a RDD
        JavaRDD<Double> rdd = sparkContext.parallelize(nums);
        Function2<Double, Double, Double> f = (a, b) -> a + b;
        Double ans = rdd.reduce(f);
        log.info("Reduced : {}", ans);
    }
}
