import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class KMeansReducer extends Reducer<IntWritable, Text, IntWritable, Text> {
    @Override
    public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        List<double[]> points = new ArrayList<>();
        for (Text value : values) {
            String[] pointStr = value.toString().split(",");
            double[] point = new double[pointStr.length];
            for (int i = 0; i < pointStr.length; i++) {
                point[i] = Double.parseDouble(pointStr[i]);
            }
            points.add(point);
        }

        double[] newCentroid = calculateCentroid(points);
        StringBuilder centroidStr = new StringBuilder();
        for (double v : newCentroid) {
            centroidStr.append(v).append(",");
        }
        centroidStr.setLength(centroidStr.length() - 1);

        context.write(key, new Text(centroidStr.toString()));
    }

    private double[] calculateCentroid(List<double[]> points) {
        int dimensions = points.get(0).length;
        double[] centroid = new double[dimensions];
        for (double[] point : points) {
            for (int i = 0; i < dimensions; i++) {
                centroid[i] += point[i];
            }
        }
        for (int i = 0; i < dimensions; i++) {
            centroid[i] /= points.size();
        }
        return centroid;
    }
}
