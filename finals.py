import numpy as np
import tensorflow as tf
import pickle
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell


pickle_in = open('data_set.pickle','rb')
train_x,train_y,test_x,test_y,weight,seq_length,n_sents,vocb_size,rev_vocab = pickle.load(pickle_in)



batch_size = 25
vocab_size =70

embedding_dim = 10

memory_dim = 200

if n_sents % float(batch_size) != 0:
    raise ValueError('Number of samples must be divisible with batch size')



enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                          name="inp%i" % t)
           for t in range(seq_length)]

labels = [tf.placeholder(tf.int32, shape=(None,),
                        name="labels%i" % t)
          for t in range(seq_length)]

weights= [tf.placeholder(tf.float32, shape=(None,),
                        name="labels%i" % t)
          for t in range(seq_length)]



dec_inp = ([tf.zeros_like(labels[0], dtype=np.int32, name="GO")]
           + labels[:-1])
           


prev_mem = tf.zeros((batch_size, memory_dim))

cell = rnn_cell.GRUCell(memory_dim)

dec_outputs, dec_memory = seq2seq.embedding_rnn_seq2seq(
    enc_inp, dec_inp, cell, vocab_size, vocab_size,embedding_dim)

loss = seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size)

tf.scalar_summary("loss", loss)



summary_op = tf.merge_all_summaries()

learning_rate = 0.5
momentum=0.9
optimizer = tf.train.MomentumOptimizer(learning_rate,momentum)
train_op = optimizer.minimize(loss)



with tf.Session() as sess:
     sess.run(tf.initialize_all_variables())
     
     n_epochs=35
     for epoch in range(n_epochs):
         epoch_loss = 0
         i=0
         while i < train_x.shape[1]:
              start = i
              end = i+batch_size
              batch_x = train_x[:,start:end]
              batch_y = train_y[:,start:end]
              batch_weight= weight[:,start:end]
              
              feed_dict = {enc_inp[t]: batch_x[t] for t in range(seq_length)}
              feed_dict.update({labels[t]: batch_y[t] for t in range(seq_length)})
              feed_dict.update({weights[t]: batch_weight[t] for t in range(seq_length)})
              
              _, loss_t, summary = sess.run([train_op, loss, summary_op], feed_dict)
              epoch_loss += loss_t
              i+=batch_size

         print('Epoch', epoch+1, 'completed out of',n_epochs,'loss:',epoch_loss) 

                
     #Testing
     X_test=train_x[:,98:99]
     Y_test=train_y[:,98:99]

     feed_dict = {enc_inp[t]: X_test[t] for t in range(seq_length)}
     feed_dict.update({labels[t]: Y_test[t] for t in range(seq_length)})

     dec_outputs_batch = sess.run(dec_outputs, feed_dict)
     
     

     t=X_test.flatten()
     tok=[rev_vocab.get(i) for i in t]
     
     n_token=[]
     for i in tok:
         if(i!= None):
            n_token.append(i)
 
     c = ''.join(n_token);
     print('Input_tr',c)





     o=Y_test.flatten()
     tok=[rev_vocab.get(i) for i in o]
     
     n_token=[]
     for i in tok:
         if(i!= None):
            n_token.append(i)
 
     c = ''.join(n_token);
     print('Target_tr',c)
     



     p=[logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]
     m=np.array(p)
     n=m.flatten()
     token=[rev_vocab.get(i) for i in n]
     
     n_token=[]
     for i in token:
         if(i!= None):
            n_token.append(i)
 
     c = ''.join(n_token);
     print('Output_tr',c)
     


     # Testing on dev_set



     X_test=test_x
     Y_test=test_y

     feed_dict = {enc_inp[t]: X_test[t] for t in range(seq_length)}
     feed_dict.update({labels[t]: Y_test[t] for t in range(seq_length)})

     dec_outputs_batch = sess.run(dec_outputs, feed_dict)
     
     

     t=X_test.flatten()
     tok=[rev_vocab.get(i) for i in t]
     
     n_token=[]
     for i in tok:
         if(i!= None):
            n_token.append(i)
 
     c = ''.join(n_token);
     print('Input_dev',c)
 





     o=Y_test.flatten()
     tok=[rev_vocab.get(i) for i in o]
     
     n_token=[]
     for i in tok:
         if(i!= None):
            n_token.append(i)
 
     c = ''.join(n_token);
     print('Target_dev',c)



     p=[logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]
     m=np.array(p)
     n=m.flatten()
     token=[rev_vocab.get(i) for i in n]
     
     n_token=[]
     for i in token:
         if(i!= None):
            n_token.append(i)
 
     c = ''.join(n_token);
     print('Output_dev',c)

     
         

         
         
         

     

     



        
     
        