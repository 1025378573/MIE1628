import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class KMeansMapper extends Mapper<Object, Text, IntWritable, Text> {
    private List<double[]> centroids = new ArrayList<>();

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        String[] centroidsStr = context.getConfiguration().get("centroids").split(";");
        for (String centroidStr : centroidsStr) {
            String[] values = centroidStr.split(",");
            double[] centroid = new double[values.length];
            for (int i = 0; i < values.length; i++) {
                centroid[i] = Double.parseDouble(values[i]);
            }
            centroids.add(centroid);
        }
    }

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
        String[] pointStr = value.toString().split(",");
        double[] point = new double[pointStr.length];
        for (int i = 0; i < pointStr.length; i++) {
            point[i] = Double.parseDouble(pointStr[i]);
        }

        int nearestCentroid = findNearestCentroid(point);
        context.write(new IntWritable(nearestCentroid), value);
    }

    private int findNearestCentroid(double[] point) {
        double minDistance = Double.MAX_VALUE;
        int nearestCentroid = -1;
        for (int i = 0; i < centroids.size(); i++) {
            double distance = 0;
            for (int j = 0; j < point.length; j++) {
                distance += Math.pow(point[j] - centroids.get(i)[j], 2);
            }
            if (distance < minDistance) {
                minDistance = distance;
                nearestCentroid = i;
            }
        }
        return nearestCentroid;
    }
}
