// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause
// See: https://spdx.org/licenses/

#include <message_infrastructure/csrc/core/channel_factory.h>
#include <message_infrastructure/csrc/core/multiprocessing.h>
#include <message_infrastructure/csrc/channel/dds/dds_channel.h>
#include <message_infrastructure/csrc/core/message_infrastructure_logging.h>
#include <gtest/gtest.h>

namespace message_infrastructure {

void dds_stop_fn() {
  // exit(0);
}

void dds_target_fn_a1_bound(int loop,
                        AbstractChannelPtr mp_to_a1,
                        AbstractChannelPtr a1_to_mp,
                        AbstractChannelPtr a1_to_a2,
                        AbstractChannelPtr a2_to_a1,
                        AbstractActor* actor_ptr) {
  actor_ptr->SetStopFn(dds_stop_fn);
  auto from_mp = mp_to_a1->GetRecvPort();
  from_mp->Start();
  auto to_mp   = a1_to_mp->GetSendPort();
  to_mp->Start();
  auto to_a2   = a1_to_a2->GetSendPort();
  to_a2->Start();
  auto from_a2 = a2_to_a1->GetRecvPort();
  from_a2->Start();
  while ((loop--)&&!actor_ptr->GetStatus()) {
    MetaDataPtr data = from_mp->Recv();
    (*reinterpret_cast<int64_t*>(data->mdata))++;
    to_a2->Send(data);
    free(data->mdata);
    data = from_a2->Recv();
    (*reinterpret_cast<int64_t*>(data->mdata))++;
    to_mp->Send(data);
    free(data->mdata);
  }
  from_mp->Join();
  from_a2->Join();
  to_a2->Join();
  to_mp->Join();
  while (!actor_ptr->GetStatus()) {
    helper::Sleep();
  }
}

void dds_target_fn_a2_bound(int loop,
                        AbstractChannelPtr a1_to_a2,
                        AbstractChannelPtr a2_to_a1,
                        AbstractActor* actor_ptr) {
  actor_ptr->SetStopFn(dds_stop_fn);
  auto from_a1 = a1_to_a2->GetRecvPort();
  from_a1->Start();
  auto to_a1 = a2_to_a1->GetSendPort();
  to_a1->Start();
  while ((loop--)&&!actor_ptr->GetStatus()) {
    MetaDataPtr data = from_a1->Recv();
    (*reinterpret_cast<int64_t*>(data->mdata))++;
    to_a1->Send(data);
    free(data->mdata);
  }
  from_a1->Join();
  to_a1->Join();
  while (!actor_ptr->GetStatus()) {
    helper::Sleep();
  }
}

TEST(TestDDSDelivery, DDSLoop) {
  MultiProcessing mp;
  int loop = 100000;
  AbstractChannelPtr mp_to_a1 = GetChannelFactory()
    .GetDDSChannel("mp_to_a1", DDSSHM, FASTDDSBackend, 5);
  AbstractChannelPtr a1_to_mp = GetChannelFactory()
    .GetDDSChannel("a1_to_mp", DDSSHM, FASTDDSBackend, 5);
  AbstractChannelPtr a1_to_a2 = GetChannelFactory()
    .GetDDSChannel("a1_to_a2", DDSSHM, FASTDDSBackend, 5);
  AbstractChannelPtr a2_to_a1 = GetChannelFactory()
    .GetDDSChannel("a2_to_a1", DDSSHM, FASTDDSBackend, 5);

  auto target_fn_a1 = std::bind(&dds_target_fn_a1_bound, loop,
                                mp_to_a1, a1_to_mp, a1_to_a2,
                                a2_to_a1, std::placeholders::_1);
  auto target_fn_a2 = std::bind(&dds_target_fn_a2_bound, loop, a1_to_a2,
                                a2_to_a1, std::placeholders::_1);

  int actor1 = mp.BuildActor(target_fn_a1);
  int actor2 = mp.BuildActor(target_fn_a2);

  auto to_a1   = mp_to_a1->GetSendPort();
  to_a1->Start();
  auto from_a1 = a1_to_mp->GetRecvPort();
  from_a1->Start();

  MetaDataPtr metadata = std::make_shared<MetaData>();
  int64_t *array = reinterpret_cast<int64_t*>(malloc(sizeof(int64_t)));
  *array = 1;
  int64_t dims[] = {1, 0, 0, 0, 0};
  int64_t nd = 1;

  GetMetadata(metadata, array, nd, METADATA_TYPES::LONG, dims);

  MetaDataPtr mptr;
  LAVA_DUMP(LOG_UTTEST, "main process loop: %d\n", loop);
  const clock_t start_time = std::clock();
  while (loop--) {
    to_a1->Send(metadata);
    mptr = from_a1->Recv();
    metadata = mptr;
  }
  from_a1->Join();
  to_a1->Join();
  mp.Stop(true);
}

TEST(TestDDSSingleProcess, DDS1Process) {
  LAVA_DUMP(LOG_UTTEST, "TestDDSSingleProcess starts.\n");
  AbstractChannelPtr dds_channel = GetChannelFactory()
    .GetDDSChannel("test_DDSChannel", DDSSHM, FASTDDSBackend, 5);

  auto send_port = dds_channel->GetSendPort();
  send_port->Start();
  auto recv_port = dds_channel->GetRecvPort();
  recv_port->Start();

  MetaDataPtr metadata = std::make_shared<MetaData>();
  metadata->nd = 1;
  metadata->type = 7;
  metadata->elsize = 8;
  metadata->total_size = 1;
  metadata->dims[0] = 1;
  metadata->strides[0] = 1;
  metadata->mdata =
    (reinterpret_cast<char*>
    (malloc(sizeof(int64_t)+sizeof(MetaData)))+sizeof(MetaData));
  *reinterpret_cast<int64_t*>(metadata->mdata) = 1;

  MetaDataPtr mptr;
  int loop = 1000;
  int i = 0;
  while (loop--) {
    if (!(loop % 1000))
      LAVA_DUMP(LOG_UTTEST, "At iteration : %d * 1000\n", i++);
    send_port->Send(metadata);
    mptr = recv_port->Recv();
    EXPECT_EQ(*reinterpret_cast<int64_t*>(mptr->mdata),
              *reinterpret_cast<int64_t*>(metadata->mdata));
    (*reinterpret_cast<int64_t*>(metadata->mdata))++;
    free(mptr->mdata);
  }
  recv_port->Join();
  send_port->Join();
}
}  // namespace message_infrastructure
