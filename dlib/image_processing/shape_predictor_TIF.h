//[TIF] Nov. 2015
//       triplet-indexing
//       version 2.0
//This code is used for the following paper:
//Heng Yang*, Renqiao Zhang*, Peter Robinson, 
//"Human and Sheep Landmarks Localisation by Triplet-Interpolated Features", WACV2016
//If you use this code please cite the above publication. 
//Part of the code is taken from dlib.net  Davis E. King (davis@dlib.net)
// The license for dlib.net is : Boost Software License   See LICENSE.txt for the full license.
#ifndef DLIB_SHAPE_PREDICToR_TIF_H_
#define DLIB_SHAPE_PREDICToR_TIF_H_

#include "shape_predictor_abstract.h"
#include "full_object_detection.h"
#include "../algs.h"
#include "../matrix.h"
#include "../geometry.h"
#include "../pixel.h"

namespace dlib
{

// ----------------------------------------------------------------------------------------

    namespace impl
    {
       
    // ------------------------------------------------------------------------------------

        class index_feature
        {
        public:

            inline std::vector<unsigned long> anchor(unsigned long id)
            {
                std::vector<unsigned long> id_xyz;
                id_xyz.push_back(anchor_idx[id]);
                id_xyz.push_back(anchor_idy[id]);
                id_xyz.push_back(anchor_idz[id]);
                return id_xyz;
            }

            inline std::vector<double> ratio (unsigned long id)
            {
                std::vector<double> ratio_out;
                ratio_out.push_back(ratio_a[id]);
                ratio_out.push_back(ratio_b[id]);
                return ratio_out;
            }

            inline unsigned long get_num_of_anchors()
            const
            {
                return anchor_idx.size();
            }

            void assign (
                const unsigned long i, 
                const unsigned long id0, const unsigned long id1, const unsigned long id2, 
                const double ratio1, const double ratio2 )
            {
                anchor_idx[i] = id0;
                anchor_idy[i] = id1;
                anchor_idz[i] = id2;
                ratio_a[i] = ratio1;
                ratio_b[i] = ratio2;
            }

            void set_size (unsigned long newsize)
            {
                anchor_idx.resize (newsize);
                anchor_idy.resize (newsize);
                anchor_idz.resize (newsize);
                ratio_a.resize (newsize);
                ratio_b.resize (newsize);
            }

            inline dlib::vector<float, 2> p_location (const matrix<float, 0,1>& shape, const unsigned long i)
            const
            {
                unsigned long idx = anchor_idx[i];
                unsigned long idy = anchor_idy[i];
                unsigned long idz = anchor_idz[i];
                double a = ratio_a[i];      double b = ratio_b[i];
                dlib::vector<float,2> p_coords;
        
                p_coords.x() = a*(location(shape,idy)[0]-location(shape,idx)[0]) + b*(location(shape,idz)[0]-location(shape,idx)[0]) + location(shape, idx)[0];
                p_coords.y() = a*(location(shape,idy)[1]-location(shape,idx)[1]) + b*(location(shape,idz)[1]-location(shape,idx)[1]) + location(shape, idx)[1];

                return p_coords;
            }

            friend inline void serialize (const index_feature& item, std::ostream& out)
            {
                dlib::serialize(item.anchor_idx, out);
                dlib::serialize(item.anchor_idy, out);
                dlib::serialize(item.anchor_idz, out);
                dlib::serialize(item.ratio_a, out);
                dlib::serialize(item.ratio_b, out);
            }

            friend inline void deserialize (index_feature& item, std::istream& in)
            {
                dlib::deserialize(item.anchor_idx, in);
                dlib::deserialize(item.anchor_idy, in);
                dlib::deserialize(item.anchor_idz, in);
                dlib::deserialize(item.ratio_a, in);
                dlib::deserialize(item.ratio_b, in);
            }


        private:
            std::vector<unsigned long> anchor_idx;
            std::vector<unsigned long> anchor_idy;
            std::vector<unsigned long> anchor_idz;
            std::vector<double> ratio_a;
            std::vector<double> ratio_b;

            inline std::vector<float> location (
                const matrix<float, 0,1>& shape,
                unsigned long idx) const
            {
                std::vector<float> loc;
                loc.push_back(  shape(idx*2)  );
                loc.push_back( shape(idx*2+1) );
                return loc;
            }

        };

    // ------------------------------------------------------------------------------------

        template <typename image_type>
        void extract_feature_pixel_values (
            //[ANDY] input>>
            const image_type& img_,
            const rectangle& rect,
            const matrix<float,0,1>& current_shape,

            const index_feature& index,

            //[ANDY] output<<
            std::vector<float>& feature_pixel_values
        )
        //[ANDY] extract the values of one single image/box
        //       columns of ref_anchor_idx and ratio correspond to feature_pixel_values by the parallel column/array index

        /*!
            requires
                - image_type == an image object that implements the interface defined in
                  dlib/image_processing/generic_image.h 
                - reference_pixel_anchor_idx.size() == reference_pixel_deltas.size()
                - current_shape.size() == reference_shape.size()
                - reference_shape.size()%2 == 0
                - max(mat(reference_pixel_anchor_idx)) < reference_shape.size()/2
            ensures
                - #feature_pixel_values.size() == reference_pixel_deltas.size()
                - for all valid i:
                    - #feature_pixel_values[i] == the value of the pixel in img_ that
                      corresponds to the pixel identified by reference_pixel_anchor_idx[i]
                      and reference_pixel_deltas[i] when the pixel is located relative to
                      current_shape rather than reference_shape.
        !*/
        {
            //[ANDY] tform_to_img: shape -> rect (img)
            const point_transform_affine tform_to_img = unnormalizing_tform(rect);

            const rectangle area = get_rect(img_);
            const_image_view<image_type> img(img_);

            //[ANDY] #_of_feature_pixels = #_of_refAnchorIDX_triplets = #_of_ratio_pairs 
            feature_pixel_values.resize( index.get_num_of_anchors() );

            for (unsigned long i = 0; i < feature_pixel_values.size(); ++i)
            {
                // Compute the point in the current shape corresponding to the i-th pixel and
                // then map it from the normalized shape space into pixel space.
                point p = tform_to_img( index.p_location(current_shape, i) );
                if (area.contains(p))
                    feature_pixel_values[i] = get_pixel_intensity( img [p.y()][p.x()] );
                else
                    feature_pixel_values[i] = 0;
            }
        }

    } // end namespace impl

// ----------------------------------------------------------------------------------------

    class shape_predictor_TIF
    {
    public:


        shape_predictor_TIF (
        ) 
        {}

        shape_predictor_TIF (
            const matrix<float,0,1>& initial_shape_,
            const std::vector<std::vector<impl::regression_tree> >& forests_,
            const std::vector<impl::index_feature> index_
        ) : initial_shape(initial_shape_), forests(forests_), index(index_)

        //[ANDY] this constructor generates a shape_predictor_TIF,
        //       consisting forests/initial_shape/anchor_idx/ratio

        /*!
            requires
                - initial_shape.size()%2 == 0
                - forests.size() == pixel_coordinates.size() == the number of cascades
                - for all valid i:
                    - all the index values in forests[i] are less than pixel_coordinates[i].size()
                - for all valid i and j: 
                    - forests[i][j].leaf_values.size() is a power of 2.
                      (i.e. we require a tree with all the levels fully filled out.
                    - forests[i][j].leaf_values.size() == forests[i][j].splits.size()+1
                      (i.e. there need to be the right number of leaves given the number of splits in the tree)
        !*/
        {}

        unsigned long num_parts (
        ) const
        {
            std::cout << "test" << std::endl;
            return initial_shape.size()/2;
        }
        
        // CTH: maintain original rect only defintion
        template <typename image_type>
        full_object_detection operator()(
             const image_type& img,
             const rectangle& rect
        ) const
        {
            return (*this)(img, rect, initial_shape);
        }
        
        // CTH: new operator which takes full_object_detection
        template <typename image_type>
        full_object_detection operator()(
           const image_type& img,
           const rectangle& rect,
           const full_object_detection& detection
        ) const
        {
            matrix<float,0,1> shape;
            matrix<float,0,1> present;
            object_to_shape(detection, shape, present);
            // could use detection.get_rect() here instead, but use annotated box as with default method
            return (*this)(img, rect, shape);
        }

        // CTH: add initial shape to default operator
        template <typename image_type>
        full_object_detection operator()(
            const image_type& img,
            const rectangle& rect,
            const matrix<float,0,1>& initial_shape_
        ) const
        {
            using namespace impl;
            matrix<float,0,1> current_shape = initial_shape_;
            std::vector<float> feature_pixel_values;

            //[ANDY] iter->cascade, i->individual trees
            for (unsigned long iter = 0; iter < forests.size(); ++iter)
            {
                extract_feature_pixel_values(img, rect, current_shape, index[iter], feature_pixel_values);
                // evaluate all the trees at this level of the cascade.
                unsigned long a;
                for (unsigned long i = 0; i < forests[iter].size(); ++i)
                    current_shape += forests[iter][i](feature_pixel_values, a);
            }

            // convert the current_shape into a full_object_detection
            const point_transform_affine tform_to_img = unnormalizing_tform(rect);

            std::vector<point> parts(current_shape.size()/2);

            for (unsigned long i = 0; i < parts.size(); ++i)
                parts[i] = tform_to_img(location(current_shape, i));
            return full_object_detection(rect, parts);
        }

        friend void serialize (const shape_predictor_TIF& item, std::ostream& out)
        {
            int version = 1;
            dlib::serialize(version, out);
            dlib::serialize(item.initial_shape, out);
            dlib::serialize(item.forests, out);
            dlib::serialize(item.index, out);
        }
        friend void deserialize (shape_predictor_TIF& item, std::istream& in)
        {
            int version = 0;
            dlib::deserialize(version, in);
            if (version != 1)
                throw serialization_error("Unexpected version found while deserializing dlib::shape_predictor_TIF.");
            dlib::deserialize(item.initial_shape, in);
            dlib::deserialize(item.forests, in);
            dlib::deserialize(item.index, in);
        }

    private:
        matrix<float,0,1> initial_shape;
        std::vector< std::vector<impl::regression_tree> > forests;
        std::vector< impl::index_feature > index;
        
        static void object_to_shape (
             const full_object_detection& obj,
             matrix<float,0,1>& shape,
             matrix<float,0,1>& present // a mask telling which elements of #shape are present.
        )
        {
            shape.set_size(obj.num_parts()*2);
            present.set_size(obj.num_parts()*2);
            const point_transform_affine tform_from_img = impl::normalizing_tform(obj.get_rect());
            for (unsigned long i = 0; i < obj.num_parts(); ++i)
            {
                if (obj.part(i) != OBJECT_PART_NOT_PRESENT)
                {
                    vector<float,2> p = tform_from_img(obj.part(i));
                    shape(2*i)   = p.x();
                    shape(2*i+1) = p.y();
                    present(2*i)   = 1;
                    present(2*i+1) = 1;
                    
                    if (length(p) > 100)
                    {
                        std::cout << "Warning, one of your objects has parts that are way outside its bounding box!  This is probably an error in your annotation." << std::endl;
                    }
                }
                else
                {
                    shape(2*i)   = 0;
                    shape(2*i+1) = 0;
                    present(2*i)   = 0;
                    present(2*i+1) = 0;
                }
            }
        }
    };
    
    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------
    // ----------------------------------------------------------------------------------------
    
    template <
        typename image_array
    >
    double test_shape_predictor_TIF (
        const shape_predictor_TIF& sp,
        const image_array& images,
        const std::vector<std::vector<full_object_detection> >& objects,
        const std::vector<std::vector<double> >& scales
    )
    {
        // make sure requires clause is not broken
#ifdef ENABLE_ASSERTS
        DLIB_CASSERT( images.size() == objects.size() ,
                     "\t double test_shape_predictor_TIF()"
                     << "\n\t Invalid inputs were given to this function. "
                     << "\n\t images.size():  " << images.size()
                     << "\n\t objects.size(): " << objects.size()
                     );
        for (unsigned long i = 0; i < objects.size(); ++i)
        {
            for (unsigned long j = 0; j < objects[i].size(); ++j)
            {
                DLIB_CASSERT(objects[i][j].num_parts() == sp.num_parts(),
                             "\t double test_shape_predictor_TIF()"
                             << "\n\t Invalid inputs were given to this function. "
                             << "\n\t objects["<<i<<"]["<<j<<"].num_parts(): " << objects[i][j].num_parts()
                             << "\n\t sp.num_parts(): " << sp.num_parts()
                             );
            }
            if (scales.size() != 0)
            {
                DLIB_CASSERT(objects[i].size() == scales[i].size(),
                             "\t double test_shape_predictor_TIF()"
                             << "\n\t Invalid inputs were given to this function. "
                             << "\n\t objects["<<i<<"].size(): " << objects[i].size()
                             << "\n\t scales["<<i<<"].size(): " << scales[i].size()
                             );
                
            }
        }
#endif
        
        running_stats<double> rs;
        for (unsigned long i = 0; i < objects.size(); ++i)
        {
            for (unsigned long j = 0; j < objects[i].size(); ++j)
            {
                // Just use a scale of 1 (i.e. no scale at all) if the caller didn't supply
                // any scales.
                const double scale = scales.size()==0 ? 1 : scales[i][j];
                
                full_object_detection det = sp(images[i], objects[i][j].get_rect());
                
                for (unsigned long k = 0; k < det.num_parts(); ++k)
                {
                    if (objects[i][j].part(k) != OBJECT_PART_NOT_PRESENT)
                    {
                        double score = length(det.part(k) - objects[i][j].part(k))/scale;
                        rs.add(score);
                    }
                }
            }
        }
        return rs.mean();
    }
    
    // ----------------------------------------------------------------------------------------
    
    template <
        typename image_array
    >
    double test_shape_predictor_TIF (
        const shape_predictor_TIF& sp,
        const image_array& images,
        const std::vector<std::vector<full_object_detection> >& objects
    )
    {
        std::vector<std::vector<double> > no_scales;
        return test_shape_predictor_TIF(sp, images, objects, no_scales);
    }
    
    // ----------------------------------------------------------------------------------------

}

#endif // DLIB_SHAPE_PREDICToR_TIF_H_

