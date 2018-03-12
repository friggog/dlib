// Copyright (C) 2014  Davis E. King (davis@dlib.net)
// License: Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SHAPE_PREDICTOR_TIF_H__
#define DLIB_SHAPE_PREDICTOR_TIF_H__

#include "dlib/string.h"
#include "dlib/geometry.h"
#include "dlib/data_io/load_image_dataset.h"
#include "dlib/image_processing.h"

using namespace std;

namespace dlib
{

// ----------------------------------------------------------------------------------------
    
    template <typename image_array>
    inline shape_predictor_TIF train_shape_predictor_TIF_on_images (
        image_array& images,
        std::vector<std::vector<full_object_detection> >& detections,
        const shape_predictor_training_options& options
    )
    {
        if (options.lambda_param <= 0)
            throw error("Invalid lambda_param value given to train_shape_predictor(), lambda_param must be > 0.");
        if (!(0 < options.nu && options.nu <= 1))
            throw error("Invalid nu value given to train_shape_predictor(). It is required that 0 < nu <= 1.");
        if (options.feature_pool_region_padding <= -0.5)
            throw error("Invalid feature_pool_region_padding value given to train_shape_predictor(), feature_pool_region_padding must be > -0.5.");

        if (images.size() != detections.size())
            throw error("The list of images must have the same length as the list of detections.");

        if (!impl::contains_any_detections(detections))
            throw error("Error, the training dataset does not have any labeled object detections in it.");

        shape_predictor_TIF_trainer trainer;

        trainer.set_cascade_depth(options.cascade_depth);
        trainer.set_tree_depth(options.tree_depth);
        trainer.set_num_trees_per_cascade_level(options.num_trees_per_cascade_level);
        trainer.set_nu(options.nu);
        trainer.set_random_seed(options.random_seed);
        trainer.set_oversampling_amount(options.oversampling_amount);
        trainer.set_feature_pool_size(options.feature_pool_size);
        trainer.set_feature_pool_region_padding(options.feature_pool_region_padding);
        trainer.set_lambda(options.lambda_param);
        trainer.set_num_test_splits(options.num_test_splits);

        if (options.be_verbose)
        {
            std::cout << "Training with cascade depth: " << options.cascade_depth << std::endl;
            std::cout << "Training with tree depth: " << options.tree_depth << std::endl;
            std::cout << "Training with " << options.num_trees_per_cascade_level << " trees per cascade level."<< std::endl;
            std::cout << "Training with nu: " << options.nu << std::endl;
            std::cout << "Training with random seed: " << options.random_seed << std::endl;
            std::cout << "Training with oversampling amount: " << options.oversampling_amount << std::endl;
            std::cout << "Training with feature pool size: " << options.feature_pool_size << std::endl;
            std::cout << "Training with feature pool region padding: " << options.feature_pool_region_padding << std::endl;
            std::cout << "Training with lambda_param: " << options.lambda_param << std::endl;
            std::cout << "Training with " << options.num_test_splits << " split tests."<< std::endl;
            trainer.be_verbose();
        }

        shape_predictor_TIF predictor = trainer.train(images, detections);

        return predictor;
    }

    inline void train_shape_predictor_TIF (
        const std::string& dataset_filename,
        const std::string& predictor_output_filename,
        const shape_predictor_training_options& options
    )
    {
        dlib::array<array2d<unsigned char> > images;
        std::vector<std::vector<full_object_detection> > objects;
        load_image_dataset(images, objects, dataset_filename);

        shape_predictor_TIF predictor = train_shape_predictor_TIF_on_images(images, objects, options);

        serialize(predictor_output_filename) << predictor;

        if (options.be_verbose)
            std::cout << "Training complete, saved predictor to file " << predictor_output_filename << std::endl;
    }

// ----------------------------------------------------------------------------------------

    template <typename image_array>
    inline double test_shape_predictor_TIF_with_images (
            image_array& images,
            std::vector<std::vector<full_object_detection> >& detections,
            std::vector<std::vector<double> >& scales,
            const shape_predictor_TIF& predictor
    )
    {
        if (images.size() != detections.size())
            throw error("The list of images must have the same length as the list of detections.");
        if (scales.size() > 0  && scales.size() != images.size())
            throw error("The list of scales must have the same length as the list of detections.");

        if (scales.size() > 0)
            return test_shape_predictor_TIF(predictor, images, detections, scales);
        else
            return test_shape_predictor_TIF(predictor, images, detections);
    }

    inline double test_shape_predictor_TIF_py (
        const std::string& dataset_filename,
        const std::string& predictor_filename
    )
    {
        // Load the images, no scales can be provided
        dlib::array<array2d<unsigned char> > images;
        // This interface cannot take the scales parameter.
        std::vector<std::vector<double> > scales;
        std::vector<std::vector<full_object_detection> > objects;
        load_image_dataset(images, objects, dataset_filename);

        // Load the shape predictor
        shape_predictor_TIF predictor;
        deserialize(predictor_filename) >> predictor;

        return test_shape_predictor_TIF_with_images(images, objects, scales, predictor);
    }

// ----------------------------------------------------------------------------------------

}

#endif // DLIB_SHAPE_PREDICTOR_TIF_H__

