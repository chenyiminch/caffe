#include <boost/thread.hpp>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

#include "caffe/common.hpp"
#include "caffe/image_data_reader.hpp"
//#include "caffe/layers/annotated_data_layer.hpp"
//#include "caffe/layers/data_layer.hpp"
//#include "caffe/layers/annotated_image_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

using namespace std;

namespace caffe {

using boost::weak_ptr;

// It has to explicitly initialize the map<> in order to work. It seems to be a
// gcc bug.
// http://www.cplusplus.com/forum/beginner/31576/

// template <>
// map<const string, weak_ptr<ImageDataReader<Datum>::Body> >
//   ImageDataReader<Datum>::bodies_
//   = map<const string, weak_ptr<ImageDataReader<Datum>::Body> >();


map<const string, weak_ptr<ImageDataReader::Body> > ImageDataReader::bodies_;
static boost::mutex bodies_mutex_;

ImageDataReader::ImageDataReader(const LayerParameter& param)
    : queue_pair_(new QueuePair(  //
        param.annotated_data_param().prefetch() * param.annotated_data_param().batch_size())) {
  // Get or create a body
  boost::mutex::scoped_lock lock(bodies_mutex_);
  string key = source_key(param);
  weak_ptr<Body>& weak = bodies_[key];
  body_ = weak.lock();
  if (!body_) {
    body_.reset(new Body(param));
    bodies_[key] = weak_ptr<Body>(body_);
  }
  body_->new_queue_pairs_.push(queue_pair_);
}

ImageDataReader::~ImageDataReader() {
  string key = source_key(body_->param_);
  body_.reset();
  boost::mutex::scoped_lock lock(bodies_mutex_);
  if (bodies_[key].expired()) {
    bodies_.erase(key);
  }
}

ImageDataReader::QueuePair::QueuePair(int size) {
  // Initialize the free queue with requested number of data
  for (int i = 0; i < size; ++i) {
    free_.push(new AnnotatedDatum());
  }
}

ImageDataReader::QueuePair::~QueuePair() {
  AnnotatedDatum* t;
  while (free_.try_pop(&t)) {
    delete t;
  }
  while (full_.try_pop(&t)) {
    delete t;
  }
}

ImageDataReader::Body::Body(const LayerParameter& param)
    : param_(param),
      new_queue_pairs_() {
  StartInternalThread();
}

ImageDataReader::Body::~Body() {
  StopInternalThread();
}

void ImageDataReader::Body::load_image_list(string image_list_fn, string root_folder, 
  vector<pair<string, string> >& ptvec_image_annotation)
{
  LOG(INFO)<<"root folder: "<<root_folder;
  if (!ptvec_image_annotation.empty())
  {
    ptvec_image_annotation.clear();
  }

  
  LOG(INFO)<<"list file: "<<image_list_fn;

  std::ifstream infile(image_list_fn.c_str());

  string line;
  size_t pos;

  while(std::getline(infile, line))
  {
    pos = line.find_last_of(' ');
    string img_fn = root_folder + line.substr(0, pos);
    string label_fn = root_folder + line.substr(pos+1);
    ptvec_image_annotation.push_back(make_pair(img_fn, label_fn));
  }

  infile.close();

  return;
}

void ImageDataReader::Body::InternalThreadEntry() {
  //shared_ptr<db::DB> db(db::GetDB(param_.data_param().backend()));
  //db->Open(param_.data_param().source(), db::READ);
  //shared_ptr<db::Cursor> cursor(db->NewCursor());
  vector<pair<string, string> > ptvec_image_annotation;
  load_image_list(param_.annotated_data_param().source(), param_.annotated_data_param().root_folder(), ptvec_image_annotation);
  vector<pair<string, string> >::iterator iter = ptvec_image_annotation.begin();

  map<string, int> name_to_label;
  string label_map_file = param_.annotated_data_param().label_map_file();
  LabelMap label_map;

  // LOG(INFO)<<"label map file: "<<label_map_file;

  CHECK(ReadProtoFromTextFile(label_map_file, &label_map))<<"Fail to read label map file.";
  CHECK(MapNameToLabel(label_map, false, &name_to_label))<<"Fail to convert name to label.";

  vector<shared_ptr<QueuePair> > qps;
  try {
    int solver_count = param_.phase() == TRAIN ? Caffe::solver_count() : 1;

    // To ensure deterministic runs, only start running once all solvers
    // are ready. But solvers need to peek on one item during initialization,
    // so read one item, then wait for the next solver.
    for (int i = 0; i < solver_count; ++i) {
      shared_ptr<QueuePair> qp(new_queue_pairs_.pop());
      //read_one(cursor.get(), qp.get());
      read_one(iter, ptvec_image_annotation, name_to_label, qp.get());
      qps.push_back(qp);
    }
    // Main loop
    while (!must_stop()) {
      for (int i = 0; i < solver_count; ++i) {
        //read_one(cursor.get(), qps[i].get());
        read_one(iter, ptvec_image_annotation, name_to_label, qps[i].get());
      }
      // Check no additional readers have been created. This can happen if
      // more than one net is trained at a time per process, whether single
      // or multi solver. It might also happen if two data layers have same
      // name and same source.
      CHECK_EQ(new_queue_pairs_.size(), 0);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
}

void ImageDataReader::Body::read_one(vector<pair<string, string> >::iterator& iter, 
  vector<pair<string, string> > ptvec_image_annotation, map<string, int> name_to_label, 
  QueuePair* qp) {
  AnnotatedDatum* anno_datum = qp->free_.pop();
  // TODO deserialize in-place instead of copy?
  // t->ParseFromString(cursor->value());

  pair<string, string> img_anno_pair = *iter;

  // LOG(INFO)<<"image file: "<<img_anno_pair.first;
  // LOG(INFO)<<"label file: "<<img_anno_pair.second; 

  bool status = ReadRichImageToAnnotatedDatum(img_anno_pair.first, 
    img_anno_pair.second, 0, 0, 0, 0, true, "jpg",
    AnnotatedDatum_AnnotationType_BBOX, "xml", 
    name_to_label, anno_datum);

  anno_datum->set_type(AnnotatedDatum_AnnotationType_BBOX);

  CHECK(status)<<"Fail to read the image and annotation.";

  qp->full_.push(anno_datum);

  // go to the next iter
  // cursor->Next();
  // if (!cursor->valid()) {
  //   DLOG(INFO) << "Restarting data prefetching from start.";
  //   cursor->SeekToFirst();
  // }

  iter++;
  if (iter == ptvec_image_annotation.end())
  {
    DLOG(INFO) << "Restarting data prefetching from start.";
    iter = ptvec_image_annotation.begin();
  }
}

// Instance class
// template class ImageDataReader<Datum>;
// template class ImageDataReader<AnnotatedDatum>;

}  // namespace caffe
