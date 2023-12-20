css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  display: flex;
  justify-content: center;
  align-items: center;
}
.chat-message.bot .avatar {
  background-color: #19C37D;
}
.chat-message.user .avatar {
  background-color: #0E1117;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
.stRadio div[role=radiogroup]{
    display: flex;
    flex-direction: row;
    justify-content: flex-start;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        Bot
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        You
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
