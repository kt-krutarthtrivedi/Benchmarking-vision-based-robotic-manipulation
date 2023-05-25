/*
Project : GRASPING OF UNKNOWN OBJECTS USING TOP SURFACES FROM A TABLE TOP
Description : This project covers a grasping system for unknown objects placed on a table. Grasp points based on object shape are calculated for each 
              individual object. 
Contributors: Ashwij Kumbla (akumbla@wpi.edu)
              Anoushka Baidya (abaidya@wpi.edu)
              Purvang Patel (pppatel@wpi.edu)
              Krutarth Ambarish Trivedi (ktrivedi@wpi.edu)
*/

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/common/centroid.h>
#include <pcl/conversions.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/boundary.h>
#include <Eigen/Core>
#include <math.h>

using std::placeholders::_1;
using Eigen::all;
using namespace std::chrono_literals;

class PointCloudProcessing
{
  public:

    pcl::PCLPointCloud2::Ptr __VoxelDownSampling__(pcl::PCLPointCloud2::Ptr CloudPtr)const
    {
      pcl::VoxelGrid<pcl::PCLPointCloud2> VoxelGridFilter;
      pcl::PCLPointCloud2::Ptr pointCloudVoxOut (new pcl::PCLPointCloud2 ());
      VoxelGridFilter.setInputCloud (CloudPtr);  
      VoxelGridFilter.setLeafSize (0.005, 0.005, 0.005);
      VoxelGridFilter.filter (*pointCloudVoxOut);
      return pointCloudVoxOut;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr __DistanceThresholding__(pcl::PCLPointCloud2::Ptr CloudPtr)const
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtr (new pcl::PointCloud<pcl::PointXYZRGB> ());
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtrFiltered (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::fromPCLPointCloud2(*CloudPtr, *xyzCloudPtr);
      pcl::PassThrough<pcl::PointXYZRGB> pass;
      pass.setInputCloud (xyzCloudPtr);
      pass.setFilterFieldName ("z");
      pass.setFilterLimits (0,1.5);
      pass.filter (*xyzCloudPtrFiltered);
      return xyzCloudPtrFiltered;   
    }

    void __RansacFiltering__(pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudPtr, 
                            pcl::ModelCoefficients::Ptr coefficients, 
                            pcl::PointIndices::Ptr inliers)const
    {
      pcl::SACSegmentation<pcl::PointXYZRGB> seg;
      seg.setOptimizeCoefficients (true);
      seg.setModelType (pcl::SACMODEL_PLANE);
      seg.setMethodType (pcl::SAC_RANSAC);
      seg.setDistanceThreshold (0.01);
      seg.setInputCloud (CloudPtr);
      seg.segment (*inliers, *coefficients);
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  __ExtractIndices__(pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudPtr,
                                                            pcl::PointIndices::Ptr inliers)const
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtrRansacFiltered (new pcl::PointCloud<pcl::PointXYZRGB> ());
      pcl::ExtractIndices<pcl::PointXYZRGB> extract;
      extract.setInputCloud (CloudPtr);
      extract.setIndices (inliers);
      extract.setNegative (true);
      extract.filter (*xyzCloudPtrRansacFiltered);
      return xyzCloudPtrRansacFiltered;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr  __ProjectInliers__(pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudPtr,
                                                            pcl::ModelCoefficients::Ptr coefficients)const
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::ProjectInliers<pcl::PointXYZRGB> proj;
      proj.setModelType (pcl::SACMODEL_PLANE);
      proj.setInputCloud (CloudPtr);
      proj.setModelCoefficients (coefficients);
      proj.filter (*cloud_projected);
      return cloud_projected;
    }

    std::vector<pcl::PointIndices> __Clustering__(pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudPtr)const
    { 
      pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB>);
      tree->setInputCloud (CloudPtr);

      std::vector<pcl::PointIndices> cluster_indices;
      pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;

      ec.setClusterTolerance (0.025); 
      ec.setMinClusterSize (10);
      ec.setMaxClusterSize (10000);
      ec.setSearchMethod (tree);
      ec.setInputCloud (CloudPtr);
      ec.extract (cluster_indices);
      return cluster_indices;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr __RemoveObjectDepth__(pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudPtr)const
    {
      pcl::PointXYZRGB minPt, maxPt;
      pcl::getMinMax3D (*CloudPtr, minPt, maxPt);
      float threshold = (std::abs(minPt.z - maxPt.z) < 0.0005) ? minPt.z : (maxPt.z - (std::abs(minPt.z - maxPt.z))*0.5);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PassThrough<pcl::PointXYZRGB> pass;
      pass.setInputCloud (CloudPtr);
      pass.setFilterFieldName ("z");
      pass.setFilterLimits (threshold, maxPt.z);
      pass.filter (*cloud_filtered);
      return cloud_filtered;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr __Projection__(pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudPtr)const
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::ModelCoefficients::Ptr coeff (new pcl::ModelCoefficients ());
      coeff->values.resize (4);
      coeff->values[0] = coeff->values[1] = 0;
      coeff->values[2] = 1.0;
      coeff->values[3] = 0;

      pcl::ProjectInliers<pcl::PointXYZRGB> project;
      project.setModelType (pcl::SACMODEL_PLANE);
      project.setInputCloud (CloudPtr);
      project.setModelCoefficients (coeff);
      project.filter (*cloud_projected);

      return cloud_projected;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr __BoundaryEstimation__(pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudPtr)const
    {
      pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
      pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
      ne.setInputCloud (CloudPtr);
      ne.setSearchMethod (pcl::search::KdTree<pcl::PointXYZRGB>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGB>));
      ne.setRadiusSearch (0.03);
      ne.compute (*normals);

      pcl::PointCloud<pcl::Boundary> boundaries;
      pcl::BoundaryEstimation<pcl::PointXYZRGB, pcl::Normal, pcl::Boundary> est;
      est.setInputCloud (CloudPtr);
      est.setInputNormals (normals);
      est.setRadiusSearch (0.02);   // 2cm radius
      est.setAngleThreshold (M_PI/1.7); 
      est.setSearchMethod (pcl::search::KdTree<pcl::PointXYZRGB>::Ptr (new pcl::search::KdTree<pcl::PointXYZRGB>));
      est.compute (boundaries);

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_boundary (new pcl::PointCloud<pcl::PointXYZRGB>);
      for(long unsigned int i = 0; i < CloudPtr->points.size(); i++) 
      { 
        if(boundaries[i].boundary_point > 0) 
        { 
	        cloud_boundary->emplace_back(CloudPtr->points[i]);
        } 
      }
      return cloud_boundary;
    }


    private:
};

class FileManagement
{
  public:
    template<typename T> 
    void __SaveFile__(T CloudPtr,std::string filename)const
    {
      pcl::PCDWriter writer;
      std::string path = dir + filename;
      writer.write(path, *CloudPtr, false); 
    }
  private:
  // CAUTION: CHANGE THE DIRECTORY
  std::string dir = "/home/purvang/Grasping-of-Unknown-Objects-using-Top-Surfaces/Media/" ;
};

class GraspQualityMatrix
{
  public:

    /********** Heuristic based Grasp Quality Metric *************/
    Eigen::Matrix< float, 4, 1 > __CloundCentre__(pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudPtr)const
    {
      Eigen::Matrix< float, 4, 1 > centroid;
      pcl::PointXYZRGB centroidpoint;

      pcl::compute3DCentroid(*CloudPtr, centroid); 
      centroidpoint.x = centroid[0];
      centroidpoint.y = centroid[1];
      centroidpoint.z = centroid[2];

      CloudPtr->push_back(centroidpoint);
      return centroid;
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

        if (dis < min_dis)
        {
          min_dis = dis;
          index_opposite_point = i;
        }
      }

      //Create a new point cloud having two grasp points only.
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr grasp_points (new pcl::PointCloud<pcl::PointXYZRGB>);

      grasp_points->push_back(CloudPtr->points[index_closest_point]);
      grasp_points->push_back(CloudPtr->points[index_opposite_point]);

      grasp_points->points[0].r = 255;
      grasp_points->points[0].g = 0;
      grasp_points->points[0].b = 0;

      grasp_points->points[1].r = 255;
      grasp_points->points[1].g = 0;
      grasp_points->points[1].b = 0;

      std::cout << "Heuristic - Point 1: " << grasp_points->points[0] << std::endl;
      std::cout << "Heuristic - Point 2: " << grasp_points->points[1] << std::endl;

      return grasp_points;
    }   
    
    /********** Geometric based Grasp Quality Metric *************/

    // Collinearty check function
    bool isCollinear(const Eigen::Vector3f& vec1, const Eigen::Vector3f& vec2, double eps = 0.1) const
    {
      // Normalize the functions before calculation dot product
      const auto& vec1_norm = vec1.normalized();
      const auto& vec2_norm = vec2.normalized();

      // derived dot product
      auto dot_product_val = vec1_norm.dot(vec2_norm);

      // making sure that dot product value is around -1 (angle as 180)
      //  since there can be numerical accuracies not giving us perfect -1
      if ((-1 - eps <= dot_product_val) && (dot_product_val <= -1 + eps))
      {
        return true;
      } 
      return false;
    }
    
    /* Function to calculate the best grasp contact pairs in the segmented point cloud */
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr
      getBestGraspContactPair(const Eigen::Matrix3Xf& normals, 
                              const Eigen::Matrix3Xf& contact_points, 
                              const Eigen::Vector3f& centroid,
                              pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudPtr
                              ) const
    { 
      
      // Initialize the containers to store the contact pairs
      std::vector<std::pair<Eigen::Vector3f,Eigen::Vector3f>> cp_pairs;

      // ideal best grasp angle value
      double best_grasp_angles = 0;

      // define angle threshold
      float angle_threshold_degree = 1;
      float angle_threshold = angle_threshold_degree * (M_PI / 180);

      // Matrix of Vectors between contact points and centroid
      auto CX0 = contact_points.colwise() - centroid;
      int index_point1;
      int index_point2;
      float min_dis = __FLT_MAX__;

      /* Check against all the contact points iteratively */
      double best_angle_clearance = 10;
      for (long int i=0; i < normals.cols(); i++)
      {
        // 1st contact point that is considered
        const auto& C1 = contact_points(all, i);

        // 1st contact point normal
        const auto& C1N = normals(all, i);

        // vector between 1st contact point and centroid
        const auto& C10 = CX0(all, i);

        for (long int j=0; j<normals.cols(); j++)
        {
          // exclude comparing between same contact points
          if (i==j)
          {
            continue;
          }

          // 2nd contact point that is considered
          const auto& C2 = contact_points(all, j);

          // vector between 2nd contact point and centroid
          const auto& C20 = CX0(all, j);

          // check if vectors between (1st contact point and centroid) and (2nd contact point and centroid) are collinear

          auto is_collinear = isCollinear(C10,C20);

          // if they are collinear check if they satisfy the necessary force vector grasp formulation
          if (is_collinear==1) //Used instead of isCollinear because we are not considering the collinearity of contact points with the centroid to calculate the best grasp 
          {

            // 1st contact point normal
            const auto& C2N = normals(all, j);
            auto C1C2 = (C1-C2).normalized();

            // calculate angles between contact points and corresponding normals
            auto angle1 = acos(C1N.dot(C1C2));
            auto angle2 = acos(C2N.dot(C1C2));
            double grasp_angle = angle1 + angle2;

            // Check if corresponding grasp angle is falling within the threshold
            if(M_PI - angle_threshold < grasp_angle && grasp_angle < M_PI + angle_threshold)
            {
               // Check if the corresponding grasp angle is better than previous candidate grasp angles
              //  if (grasp_angle >= best_grasp_angle) //correction: can we use |grasp_angle - M_PI|<best_angle_clearance instead of the current statement for getting 180deg instead of 190deg
              if(std::abs(grasp_angle - M_PI)<=best_angle_clearance) //If the grasp_angle is the closest to 180 degrees angle, then we add it to our grasp_point_cloud
               { 
                  float dist_x1 = centroid[0] - C1(0);
                  float dist_y1 = centroid[1] - C1(1);
                  float dist_z1 = centroid[2] - C1(2);

                  float dist_x2 = centroid[0] - C2(0);
                  float dist_y2 = centroid[1] - C2(1);
                  float dist_z2 = centroid[2] - C2(2);

                  float dis_point1= sqrt(dist_x1*dist_x1 + dist_y1*dist_y1 + dist_z1*dist_z1);
                  float dis_point2= sqrt(dist_x2*dist_x2 + dist_y2*dist_y2 + dist_z2*dist_z2);

                  float dis = dis_point1+ dis_point2;
                  if(dis < min_dis)
                  {
                    std::cout << "Geometric - Point 1: " << C1(0) << "," <<C1(1) <<"," << C1(2)<< std::endl;
                    std::cout << "Geometric - Point 2: " << C2(0) << "," <<C2(1) <<"," << C2(2)<< std::endl;
                    cp_pairs.push_back({C1, C2});
                    best_angle_clearance = std::abs(grasp_angle - M_PI); //the best difference from 180 degrees is updated so that it will be used for the next set of contact points 
                    best_grasp_angles = grasp_angle; //best grasp angle is updated
                    min_dis = dis;
                    index_point1 = i;
                    index_point2 = j;


                  }


               }
            }
          }
        }
      }

      //Create a new point cloud having two grasp points only.
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr grasp_points_1 (new pcl::PointCloud<pcl::PointXYZRGB>);
  
      grasp_points_1->push_back(CloudPtr->points[index_point1]);
      grasp_points_1->push_back(CloudPtr->points[index_point2]);
  
      grasp_points_1->points[0].r = 0;
      grasp_points_1->points[0].g = 255;
      grasp_points_1->points[0].b = 0;
  
      grasp_points_1->points[1].r = 0;
      grasp_points_1->points[1].g = 255;
      grasp_points_1->points[1].b = 0;

      return grasp_points_1;
    }


    
};

class PointCloudSubscriber : public rclcpp::Node
{
public:

  PointCloudSubscriber()
  : Node("PointCloudProcessor")
  {
    subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(    
      "/realsense/points", 10, std::bind(&PointCloudSubscriber::topic_callback, this, _1));

    //We know that we have 12 objects in the current simulation environment. Each original object cluster is
    //published along with the detected grasp points respectively.
    object_0 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/object_0", 10); 
    object_1 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/object_1", 10); 
    object_2 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/object_2", 10); 
    object_3 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/object_3", 10); 
    object_4 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/object_4", 10); 
    object_5 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/object_5", 10); 
    object_6 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/object_6", 10); 
    object_7 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/object_7", 10); 
    object_8 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/object_8", 10); 
    object_9 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/object_9", 10); 
    object_10 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/object_10", 10); 
    object_11 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/object_11", 10); 

    grasp_0 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_0", 10); 
    grasp_1 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_1", 10); 
    grasp_2 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_2", 10); 
    grasp_3 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_3", 10); 
    grasp_4 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_4", 10); 
    grasp_5 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_5", 10); 
    grasp_6 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_6", 10); 
    grasp_7 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_7", 10); 
    grasp_8 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_8", 10); 
    grasp_9 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_9", 10); 
    grasp_10 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_10", 10); 
    grasp_11 = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_11", 10); 

    grasp_0_G = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_0_G", 10); 
    grasp_1_G = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_1_G", 10); 
    grasp_2_G = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_2_G", 10); 
    grasp_3_G = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_3_G", 10); 
    grasp_4_G = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_4_G", 10); 
    grasp_5_G = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_5_G", 10); 
    grasp_6_G = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_6_G", 10); 
    grasp_7_G = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_7_G", 10); 
    grasp_8_G = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_8_G", 10); 
    grasp_9_G = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_9_G", 10); 
    grasp_10_G = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_10_G", 10); 
    grasp_11_G = this->create_publisher<sensor_msgs::msg::PointCloud2>("/VBRM/grasp_11_G", 10); 
  }

private:
  
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_; 

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_0;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_1;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_2;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_3;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_4;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_5;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_6;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_7;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_8;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_9;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_10;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr object_11; 

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_0;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_1;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_2;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_3;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_4;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_5;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_6;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_7;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_8;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_9;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_10;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_11; 

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_0_G;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_1_G;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_2_G;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_3_G;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_4_G;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_5_G;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_6_G;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_7_G;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_8_G;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_9_G;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_10_G;  
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr grasp_11_G; 

  PointCloudProcessing pointCloudProcessor;
  GraspQualityMatrix graspQualityMatrix;
  FileManagement fileManager;

  void topic_callback(const sensor_msgs::msg::PointCloud2::SharedPtr pointCloudIn_msg) const  
  {
    bool saveFlag = true;    //set it TRUE to save all the processed pointclouds 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr finalCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
  
    /********** Convert to PCL data type *************/
    pcl::PCLPointCloud2::Ptr pointCloud (new pcl::PCLPointCloud2 ());
    pcl_conversions::toPCL(*pointCloudIn_msg, *pointCloud);
    RCLCPP_INFO_STREAM(this->get_logger(), "Number of points in the input cloud: " << pointCloudIn_msg->width << "'"); 

    /********** Perform voxel grid downsampling filtering *************/
    pcl::PCLPointCloud2::Ptr pointCloudVoxOut (new pcl::PCLPointCloud2 ());
    pointCloudVoxOut = pointCloudProcessor.__VoxelDownSampling__(pointCloud);
    RCLCPP_INFO_STREAM(this->get_logger(), "PointCloud after Voxel Grid Downsampling: "<< pointCloudVoxOut->width * pointCloudVoxOut->height << "'"); 

    /****************** Distance Thresholding ***************/
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtrFiltered (new pcl::PointCloud<pcl::PointXYZRGB> ());
    xyzCloudPtrFiltered = pointCloudProcessor.__DistanceThresholding__(pointCloudVoxOut);
    RCLCPP_INFO_STREAM(this->get_logger(), "PointCloud after Distance Thresholding: " << xyzCloudPtrFiltered->width * xyzCloudPtrFiltered->height << "'"); 

    // /***************** RANSAC *********************/
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr xyzCloudPtrRansacFiltered (new pcl::PointCloud<pcl::PointXYZRGB> ());
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices);

    pointCloudProcessor.__RansacFiltering__(xyzCloudPtrFiltered, coefficients, inliers);
    xyzCloudPtrRansacFiltered = pointCloudProcessor.__ExtractIndices__(xyzCloudPtrFiltered, inliers);
    RCLCPP_INFO_STREAM(this->get_logger(), "PointCloud after RANSAC:  " << xyzCloudPtrRansacFiltered->size() << "'"); 

    /************************* Clustering ******************/
    std::vector<pcl::PointIndices> cluster_indices;
    cluster_indices = pointCloudProcessor.__Clustering__(xyzCloudPtrRansacFiltered);
    RCLCPP_INFO_STREAM(this->get_logger(), "Number of Objects:  " << cluster_indices.size() << "'"); 
    
    unsigned int clusterID = 0;
    for (const auto& cluster : cluster_indices)
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr objCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr objCloudTransformed (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr objCloudFlattend (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_proj (new pcl::PointCloud<pcl::PointXYZRGB>);
      
      /************************* Finding individual object ******************/
      for (const auto& idx : cluster.indices) 
      {
        objCloud->push_back((*xyzCloudPtrRansacFiltered)[idx]);
      } 
      RCLCPP_INFO_STREAM(this->get_logger(), "PointCloud representing the Cluster: " << objCloud->size () << "'"); 

      Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
      float theta = M_PI;
      transformation (0,0) = std::cos(theta);
      transformation (0,2) = std::sin(theta);
      transformation (2,0) = -std::sin(theta);
      transformation (2,2) = std::cos(theta);

      pcl::transformPointCloud (*objCloud, *objCloudTransformed, transformation);
      /************************* Finding surface for each object******************/
      objCloudFlattend = pointCloudProcessor.__RemoveObjectDepth__(objCloudTransformed);

      pcl::ModelCoefficients::Ptr cluster_coefficients (new pcl::ModelCoefficients);
      pcl::PointIndices::Ptr cluster_inliers (new pcl::PointIndices);
      pointCloudProcessor.__RansacFiltering__(objCloudFlattend, cluster_coefficients, cluster_inliers);
      cloud_proj = pointCloudProcessor.__ProjectInliers__(objCloudFlattend, cluster_coefficients);

      /************************* Projection ******************/
      
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZRGB>);
      cloud_projected = pointCloudProcessor.__Projection__(cloud_proj);
      

      /************************* Boundary Estimation ******************/
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr estimated_boundary (new pcl::PointCloud<pcl::PointXYZRGB>);
      estimated_boundary = pointCloudProcessor.__BoundaryEstimation__(cloud_projected);

      /************* Heuristic Approach For Grasp Point Detection **************************/
      /* Finding Centroid for each object */
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr centroidCloud (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::copyPointCloud(*estimated_boundary,*centroidCloud);
      const auto centroid = graspQualityMatrix.__CloundCentre__(centroidCloud);

      /* Finding the Closest Point from Centroid for each object */
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr grasp_points (new pcl::PointCloud<pcl::PointXYZRGB>);
      grasp_points = graspQualityMatrix.__FindGrasp__(centroidCloud);

      /********** Geometric Appraoch ***************/
      
      //Normal Estimation ( To Do: create a function)
      
      // Create the normal estimation class, and pass the input dataset to it
      pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_hull (new pcl::PointCloud<pcl::PointXYZ>);
      pcl::copyPointCloud(*estimated_boundary,*cloud_hull);
      pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
      ne.setInputCloud (cloud_hull);

      // Create an empty KDTree representation, and pass it to the normal estimation object.
      // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
      pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
      ne.setSearchMethod (tree);

      // Output datasets
      pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
      
      // Use all neighbors in a sphere of radius 3cm
      ne.setRadiusSearch (0.03);
      ne.useSensorOriginAsViewPoint();

      // Compute the features
      ne.compute (*cloud_normals);
      RCLCPP_INFO_STREAM(this->get_logger(), "# of normals: " << cloud_normals->size ());

      //FLIPPING NORMALS ACCORIDNG TO CENTROID
      Eigen::Matrix3Xf normal_vector_matrix(3,cloud_normals->size());
      Eigen::Matrix3Xf point_cloud(3,cloud_normals->size());
      for(size_t i = 0; i < cloud_normals->size(); i++) 
      {
        Eigen::Vector3f normal = cloud_normals->at(i).getNormalVector4fMap().head(3);
        //Eigen::Vector3f normal_dup = cloud_normals->at(i).getNormalVector4fMap().head(3);

        
        
        pcl::flipNormalTowardsViewpoint(estimated_boundary->at(i), centroid[0], centroid[1], centroid[2], normal);
        normal_vector_matrix(0,i) = normal[0];
        normal_vector_matrix(1,i) = normal[1];
        normal_vector_matrix(2,i) = normal[2];
        

        //const auto& pointMatrix = XYZcloud_filtered->at(i);
        point_cloud(0,i) = estimated_boundary->points[i].x;
        point_cloud(1,i) = estimated_boundary->points[i].y;
        point_cloud(2,i) = estimated_boundary->points[i].z;
      }

      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_vis_geo (new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::copyPointCloud(*estimated_boundary, *cloud_vis_geo);
      const auto& grasp_points_1 =  graspQualityMatrix.getBestGraspContactPair(normal_vector_matrix,point_cloud,centroid.head(3),cloud_vis_geo);
 
     

      /************* Publish Grasp Points alogn with the object cluster ******************/
      auto publish_grasp_msg = new sensor_msgs::msg::PointCloud2 ;
      pcl::PCLPointCloud2::Ptr publish_grasp(new pcl::PCLPointCloud2 ());
      pcl::toPCLPointCloud2(*grasp_points, *publish_grasp);
      pcl_conversions::fromPCL(*publish_grasp, *publish_grasp_msg);  
      publish_grasp_msg->header.frame_id = "camera_link";

      auto publish_object_msg = new sensor_msgs::msg::PointCloud2 ;
      pcl::PCLPointCloud2::Ptr publish_object(new pcl::PCLPointCloud2 ());
      pcl::toPCLPointCloud2(*objCloudTransformed, *publish_object);
      pcl_conversions::fromPCL(*publish_object, *publish_object_msg);  
      publish_object_msg->header.frame_id = "camera_link";

      //Grasp Points for Heurestic Approach
      auto publish_grasp_msg_1 = new sensor_msgs::msg::PointCloud2 ;
      pcl::PCLPointCloud2::Ptr publish_grasp_1(new pcl::PCLPointCloud2 ());
      pcl::toPCLPointCloud2(*grasp_points_1, *publish_grasp_1);
      pcl_conversions::fromPCL(*publish_grasp_1, *publish_grasp_msg_1);  
      publish_grasp_msg_1->header.frame_id = "camera_link";


       if(clusterID == 0)
      {
        object_0->publish(*publish_object_msg);
        grasp_0->publish(*publish_grasp_msg);
        grasp_0_G->publish(*publish_grasp_msg_1);
      }
      else if(clusterID == 1)
      {
        object_1->publish(*publish_object_msg);
        grasp_1->publish(*publish_grasp_msg);
        grasp_1_G->publish(*publish_grasp_msg_1);
      }
      else if(clusterID == 2)
      {
        object_2->publish(*publish_object_msg);
        grasp_2->publish(*publish_grasp_msg);
        grasp_2_G->publish(*publish_grasp_msg_1);
      }
      else if(clusterID == 3)
      {
        object_3->publish(*publish_object_msg);
        grasp_3->publish(*publish_grasp_msg);
        grasp_3_G->publish(*publish_grasp_msg_1);
      }
      else if(clusterID == 4)
      {
        object_4->publish(*publish_object_msg);
        grasp_4->publish(*publish_grasp_msg);
        grasp_4_G->publish(*publish_grasp_msg_1);
      }
      else if(clusterID == 5)
      {
        object_5->publish(*publish_object_msg);
        grasp_5->publish(*publish_grasp_msg);
        grasp_5_G->publish(*publish_grasp_msg_1);
      }
      else if(clusterID == 6)
      {
        object_6->publish(*publish_object_msg);
        grasp_6->publish(*publish_grasp_msg);
        grasp_6_G->publish(*publish_grasp_msg_1);
      }
      else if(clusterID == 7)
      {
        object_7->publish(*publish_object_msg);
        grasp_7->publish(*publish_grasp_msg);
        grasp_7_G->publish(*publish_grasp_msg_1);
      }
      else if(clusterID == 8)
      {
        object_8->publish(*publish_object_msg);
        grasp_8->publish(*publish_grasp_msg);
        grasp_8_G->publish(*publish_grasp_msg_1);
      }
      else if(clusterID == 9)
      {
        object_9->publish(*publish_object_msg);
        grasp_9->publish(*publish_grasp_msg);
        grasp_9_G->publish(*publish_grasp_msg_1);
      }
      else if(clusterID == 10)
      {
        object_10->publish(*publish_object_msg);
        grasp_10->publish(*publish_grasp_msg);
        grasp_10_G->publish(*publish_grasp_msg_1);
      }
      else if(clusterID == 11)
      {
        object_11->publish(*publish_object_msg);
        grasp_11->publish(*publish_grasp_msg);
        grasp_11_G->publish(*publish_grasp_msg_1);
      }
    
      clusterID++;
  
      /************************* Save Cloud Points as .pcd files ******************/
      if(saveFlag)
      {
        std::string objectFilename = "object_"+std::to_string(clusterID)+".pcd";
        fileManager.__SaveFile__<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>(objCloudTransformed,objectFilename);

        std::string boundaryFilename = "boundary_"+std::to_string(clusterID)+".pcd";
        fileManager.__SaveFile__<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>(estimated_boundary,boundaryFilename);

        std::string heuristicGraspPointFilename = "heuristic_grasp_point_"+std::to_string(clusterID)+".pcd";
        fileManager.__SaveFile__<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>(grasp_points,heuristicGraspPointFilename);

        std::string GeometricGraspPointFilename = "geometric_grasp_point_"+std::to_string(clusterID)+".pcd";
        fileManager.__SaveFile__<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>(grasp_points_1,GeometricGraspPointFilename);
   
      }
      *finalCloud += *estimated_boundary;
      *finalCloud += *grasp_points;
      *finalCloud += *grasp_points_1;
    }

    if(saveFlag)
    {
        std::string sceneFilename = "Scene_with_boundary_grasp_points.pcd";
        fileManager.__SaveFile__<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>(finalCloud,sceneFilename); 
    }
  }
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudSubscriber>());
  rclcpp::shutdown();
  return 0;
}
