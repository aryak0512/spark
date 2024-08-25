import org.apache.spark.ml.*;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.TrainValidationSplit;
import org.apache.spark.ml.tuning.TrainValidationSplitModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.Map;

public class HousingPriceAnalysis {

    public static void main(String[] args) {

        String path = "src/main/resources/kc_house_data.csv";
        Dataset<Row> dataset = Util.getSession().read()
                .option("header", true)
                .option("inferSchema", true) // since we need only numeric types
                .csv(path);

        dataset = dataset.withColumnsRenamed(Map.of("price","label"));

        // split into 80% training & testing data and 20% holdOut data
        Dataset<Row> [] sets = dataset.randomSplit(new double[] {0.8, 0.2});
        Dataset<Row> trainingAndTestingData = sets[0];  // test data will be for tuning the values of regParam and elasticNetParam
        Dataset<Row> holdOutData = sets[1];             // for independent validation

        // handling non-numeric fields using vectors
        StringIndexer conditionIndexer = new StringIndexer();
        conditionIndexer.setInputCol("condition");
        conditionIndexer.setOutputCol("conditionIndex");

        StringIndexer gradeIndexer = new StringIndexer();
        gradeIndexer.setInputCol("grade");
        gradeIndexer.setOutputCol("gradeIndex");

        StringIndexer zipcodeIndexer = new StringIndexer();
        zipcodeIndexer.setInputCol("zipcode");
        zipcodeIndexer.setOutputCol("zipcodeIndex");

        OneHotEncoder oneHotEncoder = new OneHotEncoder();
        oneHotEncoder.setInputCols(new String[]{"conditionIndex","gradeIndex","zipcodeIndex"});
        oneHotEncoder.setOutputCols(new String[]{"conditionVector","gradeVector","zipcodeVector"});

        // prepare the vector of features
        VectorAssembler vectorAssembler = new VectorAssembler();
        vectorAssembler.setInputCols(new String[]{"bedrooms", "bathrooms", "sqft_living", "sqft_lot","floors","grade","conditionVector","gradeVector","zipcodeVector","waterfront"});
        vectorAssembler.setOutputCol("features");

        LinearRegression linearRegression = new LinearRegression();
        // adding the regParam and elasticNetParam params
        ParamGridBuilder paramGridBuilder = new ParamGridBuilder();
        // these are some sophisticated values we add for tuning the accuracy of the model
        ParamMap[] paramMap = paramGridBuilder
                .addGrid(linearRegression.regParam(), new double[]{0.01, 0.1, 0.5})
                .addGrid(linearRegression.elasticNetParam(), new double[]{0, 0.5, 1.0})
                .build();

        TrainValidationSplit split = new TrainValidationSplit()
                .setTrainRatio(0.8)
                .setEvaluator(new RegressionEvaluator().setMetricName("r2"))
                .setEstimatorParamMaps(paramMap)
                .setEstimator(linearRegression);

        // fit and transform pipeline - a production standard way
        Pipeline pipeline = new Pipeline();
        pipeline.setStages(new PipelineStage[]{conditionIndexer, gradeIndexer, zipcodeIndexer, oneHotEncoder, vectorAssembler, split});
        PipelineModel pipelineModel = pipeline.fit(dataset);

        // run the pipeline on holdoutData and get the actual predictions
        Dataset<Row> holdOutResults = pipelineModel.transform(trainingAndTestingData);
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

        double r2Test = model.evaluate(holdOutResults).r2();
        double rmseTest = model.evaluate(holdOutResults).rootMeanSquaredError();
        System.out.println("R2 testing : " + r2Test + " | RMSE testing : " + rmseTest);

        // selecting the features for the model based on high correlation (near to 1 or -1) 0 means non-correlated
        // note : this is done using original dataset and dropping useless fields
        dataset = dataset.drop("id", "date", "waterfront", "view", "condition", "grade", "yr_renovated", "zipcode", "lat", "long");
        for ( String columnName : dataset.columns() ){
            var corr = dataset.stat().corr("label", columnName);
            System.out.println("The correlation between price and " + columnName + " is : " + corr);
        }

    }
}
