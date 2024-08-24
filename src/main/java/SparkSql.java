import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;

import static org.apache.spark.sql.functions.*;

import org.apache.spark.sql.types.*;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class SparkSql {

    private static final Logger log = LoggerFactory.getLogger(SparkSql.class);

    public static void main(String[] args) {

        SparkSession sparkSession = Util.getSession();

        // load the dataset
        String path = "src/main/resources/students.csv";
        Dataset<Row> dataset = sparkSession.read()
                .option("header", true)
                .csv(path);

//        exploreDataset(dataset);
//        exploreValues(dataset);
//        exploreFilterUsingExpressions(dataset);
//        exploreFilterUsingLambdas(dataset);
//        exploreFilterUsingFunctions(dataset);
//        exploreTemporaryViews(dataset, sparkSession);
//        exploreInMemoryDataset(sparkSession);

        var inMemoryDataset = exploreInMemoryDataset(sparkSession);
        //exploreGrouping(inMemoryDataset, sparkSession);
//        exploreFormatting(inMemoryDataset, sparkSession);
//        exploreDataFrameApi(sparkSession);
        explorePivot(sparkSession);
        sparkSession.close();
    }

    private static void explorePivot(SparkSession sparkSession) {

        Dataset<Row> dataset = sparkSession.read()
                .option("header", true)
                .csv("src/main/resources/biglog.txt");

        dataset.createOrReplaceTempView("log_data");
        dataset = sparkSession.sql("select level, date_format(datetime,'MMMM') as month from log_data");

        List<Object> columns = Arrays.asList("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December");
        Dataset<Row> pivot = dataset.groupBy(col("level")) // the Y axis label
                            .pivot(col("month"), columns) // the X axis labels
                            .count() // the aggregation function
                            .na().fill(0); // if data not available, replace by zero
        pivot.show();
    }

    /**
     * difference between dataset and dataframe is that in dataset could be of POJOs
     * e.g. Dataset<Customer>
     *
     * @param sparkSession
     */
    private static void exploreDataFrameApi(SparkSession sparkSession) {
    }

    /**
     * to find number of logs per month
     *
     * @param inMemoryDataset set to operate on
     * @param sparkSession    the session object
     * @see <p>https://spark.apache.org/docs/latest/api/sql/</p>
     */
    private static void exploreFormatting(Dataset<Row> inMemoryDataset, SparkSession sparkSession) {
        inMemoryDataset.createOrReplaceTempView("my_log_table");
        Dataset<Row> dataset = sparkSession.sql("select  date_format(message, 'MM') as d, count(1) from my_log_table group by d");
        dataset.show();
    }

    private static void exploreGrouping(Dataset<Row> dataset, SparkSession sparkSession) {

        dataset.createOrReplaceTempView("logging_view");
        Dataset<Row> dataset1 = sparkSession.sql("select level, count(message) from logging_view group by level");
        dataset1.show();
    }

    /**
     * creates an in-memory dataset, quite nasty process
     *
     * @param sparkSession
     */
    private static Dataset<Row> exploreInMemoryDataset(SparkSession sparkSession) {

        List<Row> inMemory = new ArrayList<>();
        inMemory.add(RowFactory.create("WARN", "2016-12-31 04:19:32"));
        inMemory.add(RowFactory.create("FATAL", "2016-12-31 03:22:34"));
        inMemory.add(RowFactory.create("WARN", "2016-12-31 03:21:21"));
        inMemory.add(RowFactory.create("INFO", "2015-4-21 14:32:21"));
        inMemory.add(RowFactory.create("FATAL", "2015-4-21 19:23:20"));

        // define structure of fields
        StructField[] fields = new StructField[]{
                new StructField("level", DataTypes.StringType, false, Metadata.empty()),
                new StructField("message", DataTypes.StringType, false, Metadata.empty())
        };

        StructType structType = new StructType(fields);
        // dataset and dataframe are being used interchangeably
        Dataset<Row> dataset = sparkSession.createDataFrame(inMemory, structType);
        return dataset;
    }

    private static void exploreTemporaryViews(Dataset<Row> dataset, SparkSession sparkSession) {
        // pure SQL can be run against views only
        dataset.createOrReplaceTempView("my_student_view");

        // interact with the view using sparkSession
        Dataset<Row> dataset1 = sparkSession.sql("select year , avg(score) from my_student_view group by year");
        dataset1.show();
    }

    private static void exploreFilterUsingFunctions(Dataset<Row> dataset) {
        Dataset<Row> dataset1 = dataset.filter(col("subject").like("Modern Art")
                .and(col("year").equalTo(2006)));
        dataset1.show(5);
    }

    private static void exploreFilterUsingLambdas(@NotNull Dataset<Row> dataset) {

        FilterFunction<Row> rowFilterFunction = row -> row.getAs("subject").equals("Modern Art");
        dataset.filter(rowFilterFunction).show(3);

        FilterFunction<Row> rowFilterFunction2 = row -> row.getAs("subject").equals("Modern Art")
                && Integer.parseInt(row.getAs("year")) == 2008;
        dataset.filter(rowFilterFunction2).show(3);
    }

    private static void exploreFilterUsingExpressions(@NotNull Dataset<Row> dataset) {
        Dataset<Row> modernArtRecords = dataset.filter("subject = 'Modern Art'");
        modernArtRecords.show(3);
        Dataset<Row> modernArtRecords2007 = dataset.filter("subject = 'Modern Art' and year = 2007");
        modernArtRecords2007.show(3);
    }

    private static void exploreValues(@NotNull Dataset<Row> dataset) {
        Row row = dataset.first();
        String subject = (String) row.get(2);
        int year = Integer.parseInt(row.getAs("year"));
        log.info("Year is : {} and subject is : {}", year, subject);
    }

    private static void exploreDataset(@NotNull Dataset dataset) {
        // work internally as a distributed RDD
        dataset.show(2);
        var count = dataset.count();
        log.info("Count of rows : {}", count);
    }
}
