import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.Map;

import static org.apache.spark.sql.functions.*;

public class VideoAnalysisCaseStudy {
    public static void main(String[] args) throws InterruptedException {

        // loading the dataset
        String path = "src/main/resources/casestudy/*.csv";
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
        dataset = dataset.withColumn("firstSub", when(col("firstSub").isNull(), 0).otherwise(col("firstSub")));
        dataset = dataset.filter("is_cancelled = false").drop(col("is_cancelled"));

        // split into 80% training & testing data and 20% holdOut data
        Dataset<Row> [] sets = dataset.randomSplit(new double[] {0.9, 0.1});
        Dataset<Row> trainingAndTestingData = sets[0];  // test data will be for tuning the values of regParam and elasticNetParam
        Dataset<Row> holdOutData = sets[1];             // for independent validation

        // handling non-numeric fields using vectors
        StringIndexer paymentIndexer = new StringIndexer();
        paymentIndexer.setInputCol("payment_method_type");
        paymentIndexer.setOutputCol("paymentIndex");

        StringIndexer countryIndexer = new StringIndexer();
        countryIndexer.setInputCol("country");
        countryIndexer.setOutputCol("countryIndex");

        StringIndexer periodIndexer = new StringIndexer();
        periodIndexer.setInputCol("rebill_period_in_months");
        periodIndexer.setOutputCol("periodIndex");

        OneHotEncoder oneHotEncoder = new OneHotEncoder();
        oneHotEncoder.setInputCols(new String[]{"paymentIndex","countryIndex","periodIndex"});
        oneHotEncoder.setOutputCols(new String[]{"paymentVector","countryVector","periodVector"});

        // prepare the vector of features
        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[]{"paymentVector","countryVector","periodVector","firstSub","age","all_time_views","last_month_views"});
        vectorAssembler.setOutputCol("features");

        LinearRegression linearRegression = new LinearRegression();
        // adding the regParam and elasticNetParam params
        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        // these are some sophisticated values we add for tuning the accuracy of the model
        ParamMap[] paramMap = paramGridBuilder
                .addGrid(linearRegression.regParam(), new double[]{0.01, 0.1,0.3, 0.5, 0.7, 1})
                .addGrid(linearRegression.elasticNetParam(), new double[]{0, 0.5, 1.0})
                .build();

        TrainValidationSplit split = new TrainValidationSplit()
                .setTrainRatio(0.9)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMap)
                .setEstimator(linearRegression);

        // fit and transform pipeline - a production standard way
        Pipeline pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[]{paymentIndexer, countryIndexer, periodIndexer, oneHotEncoder, vectorAssembler, split});
        PipelineModel pipelineModel = pipeline.fit(trainingAndTestingData);

        // run the pipeline on holdoutData and get the actual predictions
        Dataset<Row> holdOutResults = pipelineModel.transform(holdOutData);
        holdOutResults.show();
        holdOutResults = holdOutResults.drop("prediction");

        // pick the best model
        TrainValidationSplitModel splitModel = (TrainValidationSplitModel) pipelineModel.stages()[5];
        LinearRegressionModel model = (LinearRegressionModel) splitModel.bestModel();

        // evaluate accuracy
        // R square -> between 0 and 1, bigger the better
        // RMSE -> smaller the better
        double r2Training = model.summary().r2();
        double rmseTraining = model.summary().rootMeanSquaredError();
        System.out.println("R2 training : " + r2Training + " | RMSE training : " + rmseTraining);

        var coefficients = model.coefficients();
        var intercept = model.intercept();
        System.out.println("Intercept : " + intercept + " Coefficients : " + coefficients);

        double r2Test = model.evaluate(holdOutResults).r2();
        double rmseTest = model.evaluate(holdOutResults).rootMeanSquaredError();
        System.out.println("R2 testing : " + r2Test + " | RMSE testing : " + rmseTest);

    }
}
