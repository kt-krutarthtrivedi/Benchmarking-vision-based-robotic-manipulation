/*
Project : Cavity Detection Algorithm - An inidividual component of GRASPING OF UNKNOWN OBJECTS USING TOP SURFACES FROM A TABLE TOP
Description : The current work presents the proof of concept for the cavity detection algorithm. The idea is to pick a point cloud and explore its nearest neighbour
              and check whether its euclidean distance from the current point is within the allowed range or not. The algorithm needs to run iteratively to explore 
              outer boundary as well as all the inner boundary present. The existing development being the POC only, The future scope of this algorithm remains widely 
              open in terms of testing the same on various object geometry and its optimization.
Contributors: Krutarth Ambarish Trivedi (ktrivedi@wpi.edu); 
*/
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/features/boundary.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/passthrough.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/search/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>


/*@brief: For calculating the grasp points using the same heuristic appraoch. 
  @to-do: Fixes required in the existing algorithm as it's not working for the centroid not enclosed by the point clouds.
*/
class GraspQualityMatrix
{
  public:
    void __CloundCentre__(pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudPtr)const
    {
      Eigen::Matrix< float, 4, 1 > centroid;
      pcl::PointXYZRGB centroidpoint;

      pcl::compute3DCentroid(*CloudPtr, centroid); 
      centroidpoint.x = centroid[0];
      centroidpoint.y = centroid[1];
      centroidpoint.z = centroid[2];

      centroidpoint.r = 255;
      centroidpoint.g = 0;
      centroidpoint.b = 0;

      CloudPtr->push_back(centroidpoint);
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr __FindGrasp__(pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudPtr)const
    {
      float min_dis = FLT_MAX;
      int index_closest_point,index_opposite_point;

      for (std::size_t i = 0; i < (CloudPtr->points.size() - 1); ++i)
      {
        float dist_x = CloudPtr->points[(CloudPtr->points.size()-1)].x - CloudPtr->points[i].x;
        float dist_y = CloudPtr->points[(CloudPtr->points.size()-1)].y - CloudPtr->points[i].y;
        float dis = sqrt(dist_x*dist_x + dist_y*dist_y);

        if (dis < min_dis)
        {
          min_dis = dis;
          index_closest_point = i;
        }
      }

      pcl::PointXYZ mirrorpoint;
      mirrorpoint.x = (2*CloudPtr->points[(CloudPtr->points.size()-1)].x) - CloudPtr->points[index_closest_point].x;
      mirrorpoint.y = (2*CloudPtr->points[(CloudPtr->points.size()-1)].y) - CloudPtr->points[index_closest_point].y;
      
      for (std::size_t i = 0; i < (CloudPtr->points.size() - 1); ++i)
      {
        float dist_x = mirrorpoint.x - CloudPtr->points[i].x;
        float dist_y = mirrorpoint.y - CloudPtr->points[i].y;
        float dis = sqrt(dist_x*dist_x + dist_y*dist_y);
        std::cout << dis << " " << min_dis << std::endl;
        if (dis < min_dis)
        {
          std::cout << i << std::endl;
          min_dis = dis;
          index_opposite_point = i;
        }
      }

      std::cout << CloudPtr->points[index_closest_point] << std::endl;
      std::cout << CloudPtr->points[index_opposite_point] << std::endl;

      //Create a new point cloud having two grasp points only.
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr grasp_points (new pcl::PointCloud<pcl::PointXYZRGB>);

      grasp_points->push_back(CloudPtr->points[index_closest_point]);
      grasp_points->push_back(CloudPtr->points[index_opposite_point]);

      grasp_points->points[0].r = 0;
      grasp_points->points[0].g = 255;
      grasp_points->points[0].b = 0;

      grasp_points->points[1].r = 0;
      grasp_points->points[1].g = 255;
      grasp_points->points[1].b = 0;

      std::cout << "Heuristic - Point 1: " << grasp_points->points[0] << std::endl;
      std::cout << "Heuristic - Point 2: " << grasp_points->points[1] << std::endl;

      return grasp_points;
    }    
};

/*@brief: The novel cavity detection appraoch.
  @to-do: Optimization is required.
*/
pcl::PointCloud<pcl::PointXYZRGB>::Ptr EstimateBoundary(pcl::PointCloud<pcl::PointXYZRGB>::Ptr InputBoundary)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr OutputBoundary (new pcl::PointCloud<pcl::PointXYZRGB>);
  std::vector<pcl::PointXYZRGB> stack;
  std::vector<pcl::PointXYZRGB> data;

  // K nearest neighbor search for outer boundary
  int K = 2;
  pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
  kdtree.setInputCloud (InputBoundary);
  std::vector<int> pointIdxNKNSearch(K);
  std::vector<float> pointNKNSquaredDistance(K);

  for(int i = 0; i < InputBoundary->points.size(); i++)
  {
    pcl::PointXYZRGB searchPoint = InputBoundary->points[i];
    // std::cout << "K nearest neighbor search at (" << searchPoint.x 
    //           << " " << searchPoint.y 
    //           << " " << searchPoint.z
    //           << ") with K=" << K << std::endl;

    if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
    {
      for (std::size_t i = 0; i < pointIdxNKNSearch.size (); ++i)
      {
        // std::cout << "    "  <<   (*InputBoundary)[ pointIdxNKNSearch[i] ].x 
        //           << " " << (*InputBoundary)[ pointIdxNKNSearch[i] ].y 
        //           << " " << (*InputBoundary)[ pointIdxNKNSearch[i] ].z 
        //           << " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;

        pcl::PointXYZRGB currentPoint = (*InputBoundary)[pointIdxNKNSearch[i]];
      
        stack.push_back(currentPoint);

        if(OutputBoundary->points.size() == 0)
        {
          OutputBoundary->push_back(currentPoint);
          stack.erase(stack.begin());
          data.push_back(currentPoint);
        }

        else
        {
          int foundMatch = 0;
          int j=0;
          bool needToExplore = true;

          for (int i=0; i <data.size(); i++)
          {
            if(pcl::euclideanDistance(currentPoint,data[i]) == 0)
            {
              foundMatch = 2;
              needToExplore = false;
              break;
            }
          }

          if (needToExplore)
          {
            for(j = 0; j < stack.size(); j++) 
            { 
              if(pcl::euclideanDistance(currentPoint,stack[j]) == 0)
              {
                std::vector<double> distVec;
                for(int idx= 0; idx < OutputBoundary->points.size(); idx++) 
                {
                  distVec.push_back(pcl::euclideanDistance(currentPoint, OutputBoundary->points[idx]));
                }

                sort(distVec.begin(), distVec.end());

                if(distVec[0] == 0)
                {
                  foundMatch = 2;
                  break;
                }

                //@to-do: can be passed as an argument. 
                else if(distVec[0] < 0.01)    //0.01-Best so far! -Krutarth Trivedi 12/11/2022 
                {
                  foundMatch = 1;
                  break;
                }
              }
            }
          }

          if(foundMatch == 1)
          {
            stack.erase (stack.begin() + j);
            OutputBoundary->push_back(currentPoint);
            data.push_back(currentPoint);
          }

          else if(foundMatch == 2)
          {
            stack.erase (stack.begin() + j);
          }            
        }
      }
    }
  }

  return OutputBoundary;
}

int main (int argc, char** argv)
{
  for (unsigned int i =1; i < 2; i++)
  {
    //Input and output Paths are hard-coded now as we know the object geometry and the algorithm's outer and inner boundary
    //detection are known to us. 
    std::string inputPath = "/home/krutarth-trivedi/Grasping-of-Unknown-Objects-using-Top-Surfaces/Algorithms/Cavity_Detection/Media/object.pcd";
    std::string outputPathFiltered = "/home/krutarth-trivedi/Grasping-of-Unknown-Objects-using-Top-Surfaces/Algorithms/Cavity_Detection/Media/filtered.pcd";
    std::string outputPathBoundary = "/home/krutarth-trivedi/Grasping-of-Unknown-Objects-using-Top-Surfaces/Algorithms/Cavity_Detection/Media/boundary.pcd";
    std::string outputPathOuter = "/home/krutarth-trivedi/Grasping-of-Unknown-Objects-using-Top-Surfaces/Algorithms/Cavity_Detection/Media/outer.pcd";
    std::string outputPathInner1 = "/home/krutarth-trivedi/Grasping-of-Unknown-Objects-using-Top-Surfaces/Algorithms/Cavity_Detection/Media/inner_1.pcd";
    std::string outputPathInner2 = "/home/krutarth-trivedi/Grasping-of-Unknown-Objects-using-Top-Surfaces/Algorithms/Cavity_Detection/Media/inner_2.pcd";
    std::string outputPathPotentialSegment1 = "/home/krutarth-trivedi/Grasping-of-Unknown-Objects-using-Top-Surfaces/Algorithms/Cavity_Detection/Media/potential_segment_1.pcd";
    std::string outputPathPotentialSegment2 = "/home/krutarth-trivedi/Grasping-of-Unknown-Objects-using-Top-Surfaces/Algorithms/Cavity_Detection/potential_segment_2.pcd";
    std::string outputPathPotentialSegment3 = "/home/krutarth-trivedi/Grasping-of-Unknown-Objects-using-Top-Surfaces/Algorithms/Cavity_Detection/potential_segment_3.pcd";

    std::string outputPathGrasp1= "/home/krutarth-trivedi/Grasping-of-Unknown-Objects-using-Top-Surfaces/Algorithms/BoundaryEstimation/grasp_segment_1.pcd";
    std::string outputPathGrasp2= "/home/krutarth-trivedi/Grasping-of-Unknown-Objects-using-Top-Surfaces/Algorithms/BoundaryEstimation/grasp_segment_2.pcd";
    std::string outputPathGrasp3= "/home/krutarth-trivedi/Grasping-of-Unknown-Objects-using-Top-Surfaces/Algorithms/BoundaryEstimation/grasp_segment_3.pcd";

    pcl::PCDReader reader;
    pcl::PCDWriter writer;

    /************* Read the point cloud *****************/
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    reader.read (inputPath, *cloudptr);
        
    //Pre Processing - Remove depth (Potentially, it can be replaced with the projection algoritm implemented in the main pipeline)
    pcl::PointXYZRGB minPt, maxPt;
    pcl::getMinMax3D (*cloudptr, minPt, maxPt);
    float threshold = (std::abs(minPt.z - maxPt.z) < 0.0005) ? minPt.z : (maxPt.z - (std::abs(minPt.z - maxPt.z))*0.01);
    std::cout << "Min " << minPt.z << " Max " << maxPt.z << " Threshold " << threshold << std::endl;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud (cloudptr);
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (threshold, maxPt.z);
    pass.filter (*cloud_filtered);

    writer.write (outputPathFiltered, *cloud_filtered, false);
    
    //Estimate the surface normals
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    ne.setInputCloud (cloud_filtered);
    ne.setSearchMethod (pcl::search::KdTree<pcl::PointXYZRGB>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGB>));
    ne.setRadiusSearch (0.01);  //Simply put, if the curvature at the edge between the handle of a mug and the cylindrical part is important, the scale factor needs to be small enough to capture those details, and large otherwise.
    ne.compute (*normals);

    //Perform Boundary Estimation
    pcl::PointCloud<pcl::Boundary> boundaries;
    pcl::BoundaryEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::Boundary> est;
    est.setInputCloud (cloud_filtered);
    est.setInputNormals (normals);
    est.setRadiusSearch (0.01);   // 1cm radius
    est.setAngleThreshold (M_PI/1.9);     //Angles can be adjusted based on trial and run.
    est.setSearchMethod (pcl::search::KdTree<pcl::PointXYZRGB>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGB>));
    est.compute (boundaries);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_boundary (new pcl::PointCloud<pcl::PointXYZRGB>);
    for(int i = 0; i < cloud_filtered->points.size(); i++) 
    { 
      if(boundaries[i].boundary_point > 0) 
      { 
        cloud_boundary->emplace_back(cloud_filtered->points[i]);
      } 
    }

    writer.write (outputPathBoundary, *cloud_boundary, false);
    std::cout << "Points in the boundary cloud: " << cloud_boundary->size() << std::endl;   //502 points


    /*************** Run the cavity detection algorithm ****************/    
    //Ideally, this logical part should be designed in a way that, the algorithm runs iteratively untill it assignes each point
    //to anyone of the clusters. Right now, as the object geometry is known, some part is hard-coded.
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr outer_boundary (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr remainedPoints (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr stillRemainedPoints (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inner_boundary_1 (new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr inner_boundary_2 (new pcl::PointCloud<pcl::PointXYZRGB>);

    outer_boundary = EstimateBoundary(cloud_boundary);
    writer.write (outputPathOuter, *outer_boundary, false);
    std::cout << "Points in the outer boundary: " << outer_boundary->size() << std::endl;  
    
    //Let's search for the first cavity
    unsigned int checkFlag = cloud_boundary->points.size() - outer_boundary->points.size();
    if(checkFlag != 0)
    {  
      *remainedPoints = *cloud_boundary;
      
      for (pcl::PointCloud<pcl::PointXYZRGB>::iterator it = remainedPoints->begin(); it != remainedPoints->end(); it++) 
      {
        for (pcl::PointCloud<pcl::PointXYZRGB>::iterator it1 = outer_boundary->begin(); it1 != outer_boundary->end(); it1++) 
        {
          if(pcl::euclideanDistance(*it, *it1) == 0)
          {
            remainedPoints->erase(it);
          }
        }
      }

      // Let's filter the outliers.
      pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
      sor.setInputCloud (remainedPoints);
      sor.setMeanK (10);
      sor.setStddevMulThresh (1.0);
      sor.filter (*remainedPoints);

      //Estimate the boundary
      inner_boundary_1 = EstimateBoundary(remainedPoints);
      writer.write (outputPathInner1, *inner_boundary_1, false);
      std::cout << "Points in the inner boundary 1: " << inner_boundary_1->size() << std::endl;  
    }
    
    //Another cavity? Insetad of such linear exploration, the algorithm should explore the cavities untill checkFlag is False.
    checkFlag = remainedPoints->points.size() - inner_boundary_1->points.size();
    if(checkFlag != 0)
    {       
      *stillRemainedPoints = *remainedPoints;
      
      for (pcl::PointCloud<pcl::PointXYZRGB>::iterator it = stillRemainedPoints->begin(); it != stillRemainedPoints->end(); it++) 
      {
        for (pcl::PointCloud<pcl::PointXYZRGB>::iterator it1 = inner_boundary_1->begin(); it1 != inner_boundary_1->end(); it1++) 
        {
          if(pcl::euclideanDistance(*it, *it1) == 0)
          {
            stillRemainedPoints->erase(it);
          }
        }
      }

      // Let's filter the outliers.
      pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
      sor.setInputCloud (stillRemainedPoints);
      sor.setMeanK (10);
      sor.setStddevMulThresh (1.0);
      sor.filter (*stillRemainedPoints);

      //Estimate the boundary
      inner_boundary_2 = EstimateBoundary(stillRemainedPoints);
      writer.write (outputPathInner2, *inner_boundary_2, false);
      std::cout << "Points in the inner boundary 2: " << inner_boundary_2->size() << std::endl;  
    }
  }
  return (0);
}